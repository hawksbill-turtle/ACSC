import logging
from typing import Dict, List, Type, Union, Optional, Tuple

import ray
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.utils.typing import TensorType, ModelWeights, TensorStructType
from ray.rllib.utils.spaces.space_utils import unbatch

torch, nn = try_import_torch()
from torch.autograd import Variable

import numpy as np

import tree 

logger = logging.getLogger(__name__)

    
def normalize_(x):
    return (x - x.mean()) / (x.std() + 1e-5)


class PPOTorchPolicy(
    ValueNetworkMixin,
    LearningRateSchedule,
    EntropyCoeffSchedule,
    KLCoeffMixin,
    TorchPolicyV2,
):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.algorithms.ppo.ppo.PPOConfig().to_dict(), **config)
        # TODO: Move into Policy API, if needed at all here. Why not move this into
        #  `PPOConfig`?.
        validate_config(config)

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)

        self.EWC_task_count = 0
        self.num_task = config['num_task']
        self.task_idx = 0

        self.adversarial_loss = nn.BCELoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()

        self.device = (torch.device("cuda")
                        if torch.cuda.is_available() else torch.device("cpu"))

        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()

    @override(TorchPolicyV2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
        CL_option=None
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)
        
        curr_action_dist_ = dist_class(logits.repeat(self.num_task, 1), model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        task = train_batch['task_idx']
        task_ = np.array([num for num in range(self.num_task) for _ in range(len(train_batch['obs']))])
        task_ = Variable(torch.LongTensor(task_), requires_grad = False).to(self.device)

        valid = Variable(torch.FloatTensor(task.size(0), 1).fill_(1.0), requires_grad = False).to(self.device)
        fake = Variable(torch.FloatTensor(task.size(0), 1).fill_(0.0), requires_grad = False).to(self.device)
        valid_ = Variable(torch.FloatTensor(task_.size(0), 1).fill_(1.0), requires_grad = False).to(self.device)
        fake_ = Variable(torch.FloatTensor(task_.size(0), 1).fill_(0.0), requires_grad = False).to(self.device)

        obs_l, obs_v = model.generate_function(self.num_task, task_)

        validity, pred_label = model.discriminate_function(obs_l)
        g_loss_logit = 0.5 * (self.adversarial_loss(validity, valid_) + self.auxiliary_loss(pred_label, task_))

        validity, pred_label = model.discriminate_function(obs_v)
        g_loss_value = 0.5 * (self.adversarial_loss(validity, valid_) + self.auxiliary_loss(pred_label, task_))

        real_pred, real_aux = model.discriminate_function(train_batch['obs'])
        d_real_loss = (self.adversarial_loss(real_pred, valid) + self.auxiliary_loss(real_aux, task)) / 2

        fake_pred, fake_aux = model.discriminate_function(obs_l.detach())
        d_fake_loss_logit = (self.adversarial_loss(fake_pred, fake_) + self.auxiliary_loss(fake_aux, task_)) / 2

        fake_pred, fake_aux = model.discriminate_function(obs_v.detach())
        d_fake_loss_value = (self.adversarial_loss(fake_pred, fake_) + self.auxiliary_loss(fake_aux, task_)) / 2
        
        g_loss = g_loss_logit + g_loss_value
        d_loss = (d_real_loss + d_fake_loss_logit + d_fake_loss_value) / 3

        logits_gen_l, _ = model({'obs': obs_l})
        value_gen_l = model.value_function()
        gen_action_dist_l = dist_class(logits_gen_l, model)

        logits_gen_v, _ = model({'obs': obs_v})
        value_gen_v = model.value_function()
        gen_action_dist_v = dist_class(logits_gen_v, model)
        
        idx = torch.tensor([i for i in range(len(train_batch))]).to(self.device) + train_batch['task_idx'] * len(train_batch)

        loss1 = (nn.MSELoss()(train_batch['obs'], obs_l[idx])  \
                + nn.MSELoss()(train_batch['obs'], obs_v[idx])) / 2
        loss2 = -(normalize_(value_fn_out.repeat(self.num_task)) * normalize_(value_gen_l)   \
                + normalize_(value_fn_out.repeat(self.num_task)) * normalize_(value_gen_v)) / 2
        loss3 = (curr_action_dist_.kl(gen_action_dist_l) + curr_action_dist_.kl(gen_action_dist_v)) / 2
        
        total_loss = -surrogate_loss + self.config["vf_loss_coeff"] * vf_loss_clipped - self.entropy_coeff * curr_entropy
        total_loss = total_loss * (1 - train_batch['from_memory'])
        
        total_loss += self.config["gen_coeff"] * g_loss + self.config["dis_coeff"] * d_loss
        
        total_loss += reduce_mean_valid(self.config["etc_coeff1"] * loss1 + self.config["etc_coeff2"] * loss2 + self.config["etc_coeff3"] * loss3)
        
        if CL_option is not None and CL_option.lower() == 'ewc':
            total_loss += self.config["ewc_coeff"]*self.ewc_loss()
        
        total_loss = reduce_mean_valid(total_loss)

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_gradients_computed").
    @override(TorchPolicyV2)
    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
            }
        )

    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?
        with torch.no_grad():
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )


    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count > 0:
            losses = []
            # If "offline EWC", loop over all previous tasks
            for task in range(max(1, self.EWC_task_count+1-self.num_task), self.EWC_task_count + 1):
                for n, p in self.model.named_parameters():
                    # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                    n = n.replace('.', '__')
                    mean = getattr(self.model, '{}_EWC_prev_task{}'.format(n, task))
                    fisher = getattr(self.model, '{}_EWC_estimated_fisher{}'.format(n, task))
                    # Calculate EWC-loss
                    losses.append((fisher * (p - mean) ** 2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1. / 2) * sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return 0.


    def store_fisher_n_params(self, fisher):
        # Store new values in the network
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')
            # -mode (=MAP parameter estimate)
            self.model.register_buffer('{}_EWC_prev_task{}'.format(n, self.EWC_task_count + 1),
                                 p.detach().clone())
            self.model.register_buffer('{}_EWC_estimated_fisher{}'.format(n, self.EWC_task_count + 1),
                                 fisher[n])


    def compute_single_action_of_gan(
        self,
        obs: Optional[TensorStructType] = None,
        state: Optional[List[TensorType]] = None,
        *,
        prev_action: Optional[TensorStructType] = None,
        prev_reward: Optional[TensorStructType] = None,
        info: dict = None,
        input_dict: Optional[SampleBatch] = None,
        episode: Optional["Episode"] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        # Kwars placeholder for future compatibility.
        gen_task: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorStructType, List[TensorType], Dict[str, TensorType]]:
        """Computes and returns a single (B=1) action value.

        Takes an input dict (usually a SampleBatch) as its main data input.
        This allows for using this method in case a more complex input pattern
        (view requirements) is needed, for example when the Model requires the
        last n observations, the last m actions/rewards, or a combination
        of any of these.
        Alternatively, in case no complex inputs are required, takes a single
        `obs` values (and possibly single state values, prev-action/reward
        values, etc..).

        Args:
            obs: Single observation.
            state: List of RNN state inputs, if any.
            prev_action: Previous action value, if any.
            prev_reward: Previous reward, if any.
            info: Info object, if any.
            input_dict: A SampleBatch or input dict containing the
                single (unbatched) Tensors to compute actions. If given, it'll
                be used instead of `obs`, `state`, `prev_action|reward`, and
                `info`.
            episode: This provides access to all of the internal episode state,
                which may be useful for model-based or multi-agent algorithms.
            explore: Whether to pick an exploitation or
                exploration action
                (default: None -> use self.config["explore"]).
            timestep: The current (sampling) time step.

        Keyword Args:
            kwargs: Forward compatibility placeholder.

        Returns:
            Tuple consisting of the action, the list of RNN state outputs (if
            any), and a dictionary of extra features (if any).
        """
        # Build the input-dict used for the call to
        # `self.compute_actions_from_input_dict()`.
        if input_dict is None:
            input_dict = {SampleBatch.OBS: obs}
            if state is not None:
                for i, s in enumerate(state):
                    input_dict[f"state_in_{i}"] = s
            if prev_action is not None:
                input_dict[SampleBatch.PREV_ACTIONS] = prev_action
            if prev_reward is not None:
                input_dict[SampleBatch.PREV_REWARDS] = prev_reward
            if info is not None:
                input_dict[SampleBatch.INFOS] = info

        # Batch all data in input dict.
        input_dict = tree.map_structure_with_path(
            lambda p, s: (
                s
                if p == "seq_lens"
                else s.unsqueeze(0)
                if torch and isinstance(s, torch.Tensor)
                else np.expand_dims(s, 0)
            ),
            input_dict,
        )

        episodes = None
        if episode is not None:
            episodes = [episode]

        out = self.compute_actions_from_input_dict_from_gan(
            input_dict=SampleBatch(input_dict),
            episodes=episodes,
            explore=explore,
            timestep=timestep,
            gen_task=gen_task,
        )

        # Some policies don't return a tuple, but always just a single action.
        # E.g. ES and ARS.
        if not isinstance(out, tuple):
            single_action = out
            state_out = []
            info = {}
        # Normal case: Policy should return (action, state, info) tuple.
        else:
            l_actions, v_actions = out
            single_action = unbatch(l_actions)
        assert len(single_action) == 1
        single_action = single_action[0]

        # Return action, internal state(s), infos.
        return single_action


    def compute_actions_from_input_dict_from_gan(
        self,
        input_dict: Dict[str, TensorType],
        explore: bool = None,
        timestep: Optional[int] = None,
        gen_task: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        with torch.no_grad():
            # Pass lazy (torch) tensor dict to Model as `input_dict`.
            input_dict = self._lazy_tensor_dict(input_dict)
            input_dict.set_training(True)
            # Pack internal state inputs into (separate) list.
            state_batches = [
                input_dict[k] for k in input_dict.keys() if "state_in" in k[:8]
            ]
            # Calculate RNN sequence lengths.
            seq_lens = (
                torch.tensor(
                    [1] * len(state_batches[0]),
                    dtype=torch.long,
                    device=state_batches[0].device,
                )
                if state_batches
                else None
            )

            return self._compute_action_helper_from_gan(
                input_dict, state_batches, seq_lens, explore, timestep, gen_task
            )


    def _compute_action_helper_from_gan(
        self, input_dict, state_batches, seq_lens, explore, timestep, gen_task
    ):
        """Shared forward pass logic (w/ and w/o trajectory view API).

        Returns:
            A tuple consisting of a) actions, b) state_out, c) extra_fetches.
        """
        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep
        self._is_recurrent = state_batches is not None and state_batches != []

        # Switch to eval mode.
        if self.model:
            self.model.eval()

        task = Variable(torch.LongTensor(len(input_dict['obs'])).fill_(gen_task), requires_grad = False).to(self.device)

        self.model(input_dict, state_batches, seq_lens)
        l_gen, v_gen = self.model.generate_function(self.num_task, task)
        
        l_actions = l_gen[:, 10:17]; v_actions = v_gen[:, 10:17]

        return convert_to_numpy((l_actions, v_actions))
