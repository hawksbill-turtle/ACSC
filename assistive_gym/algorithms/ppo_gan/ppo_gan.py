"""
Proximal Policy Optimization (PPO)
==================================

This file defines the distributed Algorithm class for proximal policy
optimization.
See `ppo_[tf|torch]_policy.py` for the definition of the policy loss.

Detailed documentation: https://docs.ray.io/en/master/rllib-algorithms.html#ppo
"""

import logging
from typing import (
    Callable,
    Container,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from ray.util.debug import log_once
from ray.rllib.algorithms.algorithm import Algorithm, TrainIterCtx
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.pg import PGConfig
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
)
from assistive_gym.algorithms.algorithm_cl import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
    Deprecated,
    DEPRECATED_VALUE,
    deprecation_warning,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import ResultDict
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)

from ray.tune.logger import Logger, UnifiedLogger

from assistive_gym.algorithms.ppo_gan.ppo_gan_train_for_ewc import train_, update_fisher_
from assistive_gym.algorithms.algorithm_cl import Algorithm_CL

from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
import numpy as np

logger = logging.getLogger(__name__)

import torch
import copy
import os


class PPO_GANConfig(PGConfig):
    """Defines a configuration class from which a PPO Algorithm can be built.

    Example:
        >>> from ray.rllib.algorithms.ppo import PPO_GANConfig
        >>> config = PPO_GANConfig()  # doctest: +SKIP
        >>> config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)  # doctest: +SKIP
        >>> config = config.resources(num_gpus=0)  # doctest: +SKIP
        >>> config = config.rollouts(num_rollout_workers=4)  # doctest: +SKIP
        >>> print(config.to_dict())  # doctest: +SKIP
        >>> # Build a Algorithm object from the config and run 1 training iteration.
        >>> algo = config.build(env="CartPole-v1")  # doctest: +SKIP
        >>> algo.train()  # doctest: +SKIP

    Example:
        >>> from ray.rllib.algorithms.ppo import PPO_GANConfig
        >>> from ray import air
        >>> from ray import tune
        >>> config = PPO_GANConfig()
        >>> # Print out some default values.
        >>> print(config.clip_param)  # doctest: +SKIP
        >>> # Update the config object.
        >>> config.training(  # doctest: +SKIP
        ... lr=tune.grid_search([0.001, 0.0001]), clip_param=0.2
        ... )
        >>> # Set the config object's env.
        >>> config = config.environment(env="CartPole-v1")   # doctest: +SKIP
        >>> # Use to_dict() to get the old-style python config dict
        >>> # when running with tune.
        >>> tune.Tuner(  # doctest: +SKIP
        ...     "PPO",
        ...     run_config=air.RunConfig(stop={"episode_reward_mean": 200}),
        ...     param_space=config.to_dict(),
        ... ).fit()
    """

    def __init__(self, algo_class=None):
        """Initializes a PPO_GANConfig instance."""
        super().__init__(algo_class=algo_class or PPO_GAN)

        # fmt: off
        # __sphinx_doc_begin__
        # PPO specific settings:
        self.use_critic = True
        self.use_gae = True
        self.lambda_ = 1.0
        self.kl_coeff = 0.2
        self.sgd_minibatch_size = 128
        self.num_sgd_iter = 30
        self.shuffle_sequences = True
        self.vf_loss_coeff = 1.0
        self.entropy_coeff = 0.0
        self.entropy_coeff_schedule = None
        self.clip_param = 0.3
        self.vf_clip_param = 10.0
        self.grad_clip = None
        self.kl_target = 0.01

        # Override some of PG/AlgorithmConfig's default values with PPO-specific values.
        self.num_rollout_workers = 2
        self.train_batch_size = 4000
        self.lr = 5e-5
        self.model["vf_share_layers"] = False
        self._disable_preprocessor_api = False
        # __sphinx_doc_end__
        # fmt: on

        # Deprecated keys.
        self.vf_share_layers = DEPRECATED_VALUE
        
        self.EWC_task_count = 0
        self.num_task = 0
        self.evaluation_envs = []

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        lr_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
        use_critic: Optional[bool] = NotProvided,
        use_gae: Optional[bool] = NotProvided,
        lambda_: Optional[float] = NotProvided,
        kl_coeff: Optional[float] = NotProvided,
        sgd_minibatch_size: Optional[int] = NotProvided,
        num_sgd_iter: Optional[int] = NotProvided,
        shuffle_sequences: Optional[bool] = NotProvided,
        vf_loss_coeff: Optional[float] = NotProvided,
        entropy_coeff: Optional[float] = NotProvided,
        entropy_coeff_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
        clip_param: Optional[float] = NotProvided,
        vf_clip_param: Optional[float] = NotProvided,
        grad_clip: Optional[float] = NotProvided,
        kl_target: Optional[float] = NotProvided,
        # Deprecated.
        vf_share_layers=DEPRECATED_VALUE,
        **kwargs,
    ) -> "PPO_GANConfig":
        """Sets the training related configuration.

        Args:
            lr_schedule: Learning rate schedule. In the format of
                [[timestep, lr-value], [timestep, lr-value], ...]
                Intermediary timesteps will be assigned to interpolated learning rate
                values. A schedule should normally start from timestep 0.
            use_critic: Should use a critic as a baseline (otherwise don't use value
                baseline; required for using GAE).
            use_gae: If true, use the Generalized Advantage Estimator (GAE)
                with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
            lambda_: The GAE (lambda) parameter.
            kl_coeff: Initial coefficient for KL divergence.
            sgd_minibatch_size: Total SGD batch size across all devices for SGD.
                This defines the minibatch size within each epoch.
            num_sgd_iter: Number of SGD iterations in each outer loop (i.e., number of
                epochs to execute per train batch).
            shuffle_sequences: Whether to shuffle sequences in the batch when training
                (recommended).
            vf_loss_coeff: Coefficient of the value function loss. IMPORTANT: you must
                tune this if you set vf_share_layers=True inside your model's config.
            entropy_coeff: Coefficient of the entropy regularizer.
            entropy_coeff_schedule: Decay schedule for the entropy regularizer.
            clip_param: PPO clip parameter.
            vf_clip_param: Clip param for the value function. Note that this is
                sensitive to the scale of the rewards. If your expected V is large,
                increase this.
            grad_clip: If specified, clip the global norm of gradients by this amount.
            kl_target: Target value for KL divergence.

        Returns:
            This updated AlgorithmConfig object.
        """
        if vf_share_layers != DEPRECATED_VALUE:
            deprecation_warning(
                old="ppo.DEFAULT_CONFIG['vf_share_layers']",
                new="PPO_GANConfig().training(model={'vf_share_layers': ...})",
                error=True,
            )

        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if lr_schedule is not NotProvided:
            self.lr_schedule = lr_schedule
        if use_critic is not NotProvided:
            self.use_critic = use_critic
        if use_gae is not NotProvided:
            self.use_gae = use_gae
        if lambda_ is not NotProvided:
            self.lambda_ = lambda_
        if kl_coeff is not NotProvided:
            self.kl_coeff = kl_coeff
        if sgd_minibatch_size is not NotProvided:
            self.sgd_minibatch_size = sgd_minibatch_size
        if num_sgd_iter is not NotProvided:
            self.num_sgd_iter = num_sgd_iter
        if shuffle_sequences is not NotProvided:
            self.shuffle_sequences = shuffle_sequences
        if vf_loss_coeff is not NotProvided:
            self.vf_loss_coeff = vf_loss_coeff
        if entropy_coeff is not NotProvided:
            if isinstance(entropy_coeff, int):
                entropy_coeff = float(entropy_coeff)
            if entropy_coeff < 0.0:
                raise ValueError("`entropy_coeff` must be >= 0.0")
            self.entropy_coeff = entropy_coeff
        if entropy_coeff_schedule is not NotProvided:
            self.entropy_coeff_schedule = entropy_coeff_schedule
        if clip_param is not NotProvided:
            self.clip_param = clip_param
        if vf_clip_param is not NotProvided:
            self.vf_clip_param = vf_clip_param
        if grad_clip is not NotProvided:
            self.grad_clip = grad_clip
        if kl_target is not NotProvided:
            self.kl_target = kl_target

        return self

    @override(AlgorithmConfig)
    def validate(self) -> None:
        # Call super's validation method.
        super().validate()

        # SGD minibatch size must be smaller than train_batch_size (b/c
        # we subsample a batch of `sgd_minibatch_size` from the train-batch for
        # each `num_sgd_iter`).
        # Note: Only check this if `train_batch_size` > 0 (DDPPO sets this
        # to -1 to auto-calculate the actual batch size later).
        if self.sgd_minibatch_size > self.train_batch_size:
            raise ValueError(
                f"`sgd_minibatch_size` ({self.sgd_minibatch_size}) must be <= "
                f"`train_batch_size` ({self.train_batch_size}). In PPO, the train batch"
                f" is be split into {self.sgd_minibatch_size} chunks, each of which is "
                f"iterated over (used for updating the policy) {self.num_sgd_iter} "
                "times."
            )

        # Episodes may only be truncated (and passed into PPO's
        # `postprocessing_fn`), iff generalized advantage estimation is used
        # (value function estimate at end of truncated episode to estimate
        # remaining value).
        if (
            not self.in_evaluation
            and self.batch_mode == "truncate_episodes"
            and not self.use_gae
        ):
            raise ValueError(
                "Episode truncation is not supported without a value "
                "function (to estimate the return at the end of the truncated"
                " trajectory). Consider setting "
                "batch_mode=complete_episodes."
            )


class UpdateKL:
    """Callback to update the KL based on optimization info.

    This is used inside the execution_plan function. The Policy must define
    a `update_kl` method for this to work. This is achieved for PPO via a
    Policy mixin class (which adds the `update_kl` method),
    defined in ppo_[tf|torch]_policy.py.
    """

    def __init__(self, workers):
        self.workers = workers

    def __call__(self, fetches):
        def update(pi, pi_id):
            assert LEARNER_STATS_KEY not in fetches, (
                "{} should be nested under policy id key".format(LEARNER_STATS_KEY),
                fetches,
            )
            if pi_id in fetches:
                kl = fetches[pi_id][LEARNER_STATS_KEY].get("kl")
                assert kl is not None, (fetches, pi_id)
                # Make the actual `Policy.update_kl()` call.
                pi.update_kl(kl)
            else:
                logger.warning("No data for {}, not updating kl".format(pi_id))

        # Update KL on all trainable policies within the local (trainer)
        # Worker.
        self.workers.local_worker().foreach_policy_to_train(update)


class PPO_GAN(Algorithm_CL):
    def __init__(
        self,
        config: Optional[AlgorithmConfig] = None,
        env=None,  # deprecated arg
        logger_creator: Optional[Callable[[], Logger]] = None,
        **kwargs,
    ):
        self.num_task = config['num_task']
        self.EWC_task_count = config['EWC_task_count']
        self.evaluation_envs = config['eval_envs']
        
        if config['load_memory']:
            self.memory_batch_size = config['memory_batch_size']

        super().__init__(
            config=config,
            env=env,
            logger_creator=logger_creator,
            **kwargs,
        )

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPO_GANConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            from assistive_gym.algorithms.ppo_gan.ppo_torch_policy import PPOTorchPolicy

            return PPOTorchPolicy
        elif config["framework"] == "tf":
            from assistive_gym.algorithms.ppo_gan.ppo_tf_policy import PPOTF1Policy

            return PPOTF1Policy
        else:
            from assistive_gym.algorithms.ppo_gan.ppo_tf_policy import PPOTF2Policy

            return PPOTF2Policy


    @ExperimentalAPI
    def training_step(self, CL_option=None, lock_update=False, memory_generator=None) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        if not lock_update:
            if self.config.count_steps_by == "agent_steps":
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers, max_agent_steps=self.config.train_batch_size
                )
            else:
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers, max_env_steps=self.config.train_batch_size
                )
        else:
            if self.config.count_steps_by == "agent_steps":
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers, max_agent_steps=self.config.num_workers
                )
            else:
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers, max_env_steps=self.config.num_workers
                )

        if self.config.save_memory == True and self.config.memory_root is not None:
            memory_fn = 'memory_' + self.config.env[14:-3] + '.pt'
            memory_fn = os.path.join(self.config.memory_root, memory_fn)
            if os.path.exists(memory_fn):
                previous_memory = torch.load(memory_fn)
                if len(previous_memory) < self.config.save_memory_len:
                    if self.config.save_memory_len - len(previous_memory) >= len(train_batch):
                        current_memory = previous_memory.concat_samples([previous_memory, train_batch])
                    else:
                        additional_batch = train_batch.slice(0, self.config.save_memory_len - len(previous_memory))
                        current_memory = previous_memory.concat_samples([previous_memory, additional_batch])
                    torch.save(current_memory, memory_fn)
            else:
                if len(train_batch) <= self.config.save_memory_len:
                    torch.save(train_batch, memory_fn)
                else: 
                    additional_batch = train_batch.slice(0, self.config.save_memory_len)
                    torch.save(additional_batch, memory_fn)

        train_batch['task_idx'] = np.array([self.evaluation_envs.index(self.config.env) for _ in range(len(train_batch))])
        train_batch['from_memory'] = np.array([0 for _ in range(len(train_batch))])

        if self.config.load_memory and not lock_update:
            memory_batch = next(memory_generator)
            train_batch = memory_batch.concat(train_batch).shuffle()
        
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["advantages"])
        ## Train
        if self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch, CL_option=CL_option, lock_update=lock_update)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch, CL_option=CL_option, lock_update=lock_update)

        policies_to_update = list(train_results.keys())

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: self.workers.local_worker().policy_map[pid].num_grad_updates
                for pid in policies_to_update
            },
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.num_remote_workers() > 0:
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(
                    policies=policies_to_update,
                    global_vars=global_vars,
                )

        # For each policy: Update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                log_once("ppo_warned_lr_ratio")
                and self.config.get("model", {}).get("vf_share_layers")
                and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                log_once("ppo_warned_vf_clip")
                and mean_reward > self.config.vf_clip_param
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results
    
    def train(self, CL_option=None, memory_generator=None):
        return train_(self, CL_option, memory_generator)
    
    def update_fisher(self, ewc_epochs=100, num_ewc_steps=40):
        fisher_info={}

        for n, p in self.get_policy().model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                fisher_info[n] = p.detach().clone().zero_()
        
        def set_total_rollout_fragment_length(worker, num):
            worker.total_rollout_fragment_length = num
        
        from tqdm import tqdm
        for batch in tqdm(range(ewc_epochs)):
            for step in range(num_ewc_steps):
                result, fisher_info = update_fisher_(self, fisher_info)

        for n, p in self.get_policy().model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                fisher_info[n] = fisher_info[n] / 100
        
        self.get_policy().store_fisher_n_params(fisher_info)
        print('fisher computed successfully')
    
    def set_EWC_num(self, EWC_task_count, num_task):
        self.get_policy().EWC_task_count = EWC_task_count
        self.get_policy().num_task = num_task
        self.get_policy().task_idx = EWC_task_count % num_task
    
    def add_state_dict_keys(self, EWC_task_count):
        for task in range(EWC_task_count):
            for n, p in self.get_policy().model.named_parameters():
                n = n.replace('.', '__')
                self.get_policy().model.register_buffer('{}_EWC_prev_task{}'.format(n, task+1), p.detach().clone().zero_())
                self.get_policy().model.register_buffer('{}_EWC_estimated_fisher{}'.format(n, task+1), p.detach().clone().zero_())


# Deprecated: Use ray.rllib.algorithms.ppo.PPO_GANConfig instead!
class _deprecated_default_config(dict):
    def __init__(self):
        super().__init__(PPO_GANConfig().to_dict())

    @Deprecated(
        old="assistive_gym.agents.ppo_gan.ppo_gan::DEFAULT_CONFIG",
        new="assistive_gym.algorithms.ppo_gan.ppo_gan::PPO_GANConfig(...)",
        error=True,
    )
    def __getitem__(self, item):
        return super().__getitem__(item)


def memory_minibatches(samples: SampleBatch, mem_minibatch_size: int):
    """Return a generator yielding minibatches from a sample batch.

    Args:
        samples: SampleBatch to split up.
        sgd_minibatch_size: Size of minibatches to return.
        shuffle: Whether to shuffle the order of the generated minibatches.
            Note that in case of a non-recurrent policy, the incoming batch
            is globally shuffled first regardless of this setting, before
            the minibatches are generated from it!

    Yields:
        SampleBatch: Each of size `sgd_minibatch_size`.
    """
    offset = 0
    samples.shuffle()
    while True:
        mem = samples[1:1]; target_size = mem_minibatch_size
        while target_size > 0:
            m = samples[offset:offset+min(target_size, len(samples)-offset)]
            mem = mem.concat(m)
            target_size -= min(target_size, len(samples)-offset)
            offset += len(m)
            if offset == len(samples):
                samples.shuffle()
                offset = 0
        target_size = mem_minibatch_size
        yield mem


DEFAULT_CONFIG = _deprecated_default_config()
