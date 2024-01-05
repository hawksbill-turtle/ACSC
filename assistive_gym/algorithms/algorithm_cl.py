from collections import defaultdict
import concurrent
import copy
from datetime import datetime
import functools
import gym
import importlib
import json
import logging
import numpy as np
import os
from packaging import version
import pkg_resources
import re
import tempfile
import time
import tree  # pip install dm_tree
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

import ray
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.actor import ActorHandle
from ray.air.checkpoint import Checkpoint
from ray.rllib.connectors.agent.obs_preproc import ObsPreprocessorConnector
import ray.cloudpickle as pickle
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.registry import ALGORITHMS as ALL_ALGORITHMS
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.utils import _gym_env_creator
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.metrics import (
    collect_episodes,
    collect_metrics,
    summarize_episodes,
)
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.common import (
    STEPS_TRAINED_THIS_ITER_COUNTER,  # TODO: Backward compatibility.
    LEARN_ON_BATCH_TIMER,
    LOAD_BATCH_TIMER,
)
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.offline import get_dataset_and_shards
from ray.rllib.offline.estimators import (
    OffPolicyEstimator,
    ImportanceSampling,
    WeightedImportanceSampling,
    DirectMethod,
    DoublyRobust,
)
from ray.rllib.offline.offline_evaluation_utils import remove_time_dim
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch, MultiAgentBatch, concat_samples
from ray.rllib.utils import deep_update, FilterManager, NullContextManager, force_list
from ray.rllib.utils.annotations import (
    DeveloperAPI,
    ExperimentalAPI,
    OverrideToImplementCustomLogic,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
    PublicAPI,
    override,
)
from ray.rllib.utils.checkpoints import CHECKPOINT_VERSION, get_checkpoint_info
from ray.rllib.utils.debug import summarize, update_global_seed_if_necessary
from ray.rllib.utils.deprecation import (
    DEPRECATED_VALUE,
    Deprecated,
    deprecation_warning,
)
from ray.rllib.utils.error import ERR_MSG_INVALID_ENV_DESCRIPTOR, EnvError
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.metrics import (
    DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_SAMPLED_THIS_ITER,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_THIS_ITER,
    NUM_ENV_STEPS_TRAINED,
    NUM_GRAD_UPDATES_LIFETIME,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TRAINING_ITERATION_TIMER,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.replay_buffers import MultiAgentReplayBuffer, ReplayBuffer
from ray.rllib.utils.spaces import space_utils
from ray.rllib.utils.typing import (
    AgentConnectorDataType,
    AgentID,
    AlgorithmConfigDict,
    EnvCreator,
    EnvInfoDict,
    EnvType,
    EpisodeID,
    GradInfoDict,
    ModelGradients,
    PartialAlgorithmConfigDict,
    PolicyID,
    PolicyState,
    ResultDict,
    SampleBatchType,
    TensorStructType,
    TensorType,
)
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.experiment.trial import ExportFormat
from ray.tune.logger import Logger, UnifiedLogger
from ray.tune.registry import ENV_CREATOR, _global_registry
from ray.tune.resources import Resources
from ray.tune.result import (
    DEBUG_METRICS,
    DEFAULT_RESULTS_DIR,
    DONE,
    EPISODES_THIS_ITER,
    EPISODES_TOTAL,
    HOSTNAME,
    NODE_IP,
    PID,
    RESULT_DUPLICATE,
    SHOULD_CHECKPOINT,
    STDERR_FILE,
    STDOUT_FILE,
    TIME_THIS_ITER_S,
    TIME_TOTAL_S,
    TIMESTEPS_THIS_ITER,
    TIMESTEPS_TOTAL,
    TRAINING_ITERATION,
    TRIAL_ID,
    TRIAL_INFO,
)
from ray.tune.trainable import Trainable
from ray.util import log_once
from ray.util.timer import _Timer
from ray.tune.registry import get_trainable_cls

from ray.rllib.algorithms.algorithm import Algorithm, TrainIterCtx

from ray.tune.utils import UtilMonitor
from ray.tune.trainable.trainable import SETUP_TIME_THRESHOLD

from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.rllib.utils.sgd import minibatches
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.policy.torch_policy import _directStepOptimizerSingleton

from ray.tune.trainable.util import TrainableUtil

import torch
import threading
import uuid
import math

tf1, tf, tfv = try_import_tf()

logger = logging.getLogger(__name__)


@Deprecated(
    new="config = AlgorithmConfig().update_from_dict({'a': 1, 'b': 2}); ... ; "
    "print(config.lr) -> 0.001; if config.a > 0: [do something];",
    error=False,
)
def with_common_config(extra_config):
    return Algorithm.merge_trainer_configs(
        AlgorithmConfig().to_dict(), extra_config, _allow_unknown_configs=True
    )


@PublicAPI
class Algorithm_CL(Algorithm):
    @PublicAPI
    def __init__(
        self,
        config: Optional[AlgorithmConfig] = None,
        env=None,  # deprecated arg
        logger_creator: Optional[Callable[[], Logger]] = None,
        **kwargs,
    ):
        """Initializes an Algorithm instance.

        Args:
            config: Algorithm-specific configuration object.
            logger_creator: Callable that creates a ray.tune.Logger
                object. If unspecified, a default logger is created.
            **kwargs: Arguments passed to the Trainable base class.
        """
        config = config or self.get_default_config()

        # Translate possible dict into an AlgorithmConfig object, as well as,
        # resolving generic config objects into specific ones (e.g. passing
        # an `AlgorithmConfig` super-class instance into a PPO constructor,
        # which normally would expect a PPOConfig object).
        if isinstance(config, dict):
            default_config = self.get_default_config()
            # `self.get_default_config()` also returned a dict ->
            # Last resort: Create core AlgorithmConfig from merged dicts.
            if isinstance(default_config, dict):
                config = AlgorithmConfig.from_dict(
                    config_dict=self.merge_trainer_configs(default_config, config, True)
                )
            # Default config is an AlgorithmConfig -> update its properties
            # from the given config dict.
            else:
                config = default_config.update_from_dict(config)
        else:
            default_config = self.get_default_config()
            # Given AlgorithmConfig is not of the same type as the default config:
            # This could be the case e.g. if the user is building an algo from a
            # generic AlgorithmConfig() object.
            if not isinstance(config, type(default_config)):
                config = default_config.update_from_dict(config.to_dict())

        # In case this algo is using a generic config (with no algo_class set), set it
        # here.
        if config.algo_class is None:
            config.algo_class = type(self)

        if env is not None:
            deprecation_warning(
                old=f"algo = Algorithm(env='{env}', ...)",
                new=f"algo = AlgorithmConfig().environment('{env}').build()",
                error=False,
            )
            config.environment(env)

        # Validate and freeze our AlgorithmConfig object (no more changes possible).
        config.validate()
        config.freeze()

        # Convert `env` provided in config into a concrete env creator callable, which
        # takes an EnvContext (config dict) as arg and returning an RLlib supported Env
        # type (e.g. a gym.Env).
        self._env_id, self.env_creator = self._get_env_id_and_creator(
            config.env, config
        )
        env_descr = (
            self._env_id.__name__ if isinstance(self._env_id, type) else self._env_id
        )

        # Placeholder for a local replay buffer instance.
        self.local_replay_buffer = None

        # Create a default logger creator if no logger_creator is specified
        if logger_creator is None:
            # Default logdir prefix containing the agent's name and the
            # env id.
            timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
            env_descr_for_dir = re.sub("[/\\\\]", "-", str(env_descr))
            logdir_prefix = f"{str(self)}_{env_descr_for_dir}_{timestr}"
            if not os.path.exists(DEFAULT_RESULTS_DIR):
                # Possible race condition if dir is created several times on
                # rollout workers
                os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
            logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=DEFAULT_RESULTS_DIR)

            # Allow users to more precisely configure the created logger
            # via "logger_config.type".
            if config.logger_config and "type" in config.logger_config:

                def default_logger_creator(config):
                    """Creates a custom logger with the default prefix."""
                    cfg = config["logger_config"].copy()
                    cls = cfg.pop("type")
                    # Provide default for logdir, in case the user does
                    # not specify this in the "logger_config" dict.
                    logdir_ = cfg.pop("logdir", logdir)
                    return from_config(cls=cls, _args=[cfg], logdir=logdir_)

            # If no `type` given, use tune's UnifiedLogger as last resort.
            else:

                def default_logger_creator(config):
                    """Creates a Unified logger with the default prefix."""
                    return UnifiedLogger(config, logdir, loggers=None)

            logger_creator = default_logger_creator

        # Metrics-related properties.
        self._timers = defaultdict(_Timer)
        self._counters = defaultdict(int)
        self._episode_history = []
        self._episodes_to_be_collected = []

        # The fully qualified AlgorithmConfig used for evaluation
        # (or None if evaluation not setup).
        self.evaluation_config: Optional[AlgorithmConfig] = None
        # Evaluation WorkerSet and metrics last returned by `self.evaluate()`.
        self.evaluation_workers = []
        # Initialize common evaluation_metrics to nan, before they become
        # available. We want to make sure the metrics are always present
        # (although their values may be nan), so that Tune does not complain
        # when we use these as stopping criteria.
        self.evaluation_metrics = {
            "evaluation": {
                "episode_reward_max": np.nan,
                "episode_reward_min": np.nan,
                "episode_reward_mean": np.nan,
            }
        }

        self._experiment_id = uuid.uuid4().hex
        self.config = config or {}
        trial_info = self.config.pop(TRIAL_INFO, None)

        if self.is_actor():
            disable_ipython()

        self._result_logger = self._logdir = None
        self._create_logger(self.config, logger_creator)

        self._stdout_context = self._stdout_fp = self._stdout_stream = None
        self._stderr_context = self._stderr_fp = self._stderr_stream = None
        self._stderr_logging_handler = None

        stdout_file = self.config.pop(STDOUT_FILE, None)
        stderr_file = self.config.pop(STDERR_FILE, None)
        self._open_logfiles(stdout_file, stderr_file)

        self._iteration = 0
        self._time_total = 0.0
        self._timesteps_total = None
        self._episodes_total = None
        self._time_since_restore = 0.0
        self._timesteps_since_restore = 0
        self._iterations_since_restore = 0
        self._last_result = None
        self._restored = False
        self._trial_info = trial_info
        self._stdout_file = stdout_file
        self._stderr_file = stderr_file

        start_time = time.time()
        self._local_ip = ray.util.get_node_ip_address()
        self.setup(copy.deepcopy(self.config))
        setup_time = time.time() - start_time
        if setup_time > SETUP_TIME_THRESHOLD:
            logger.info(
                "Trainable.setup took {:.3f} seconds. If your "
                "trainable is slow to initialize, consider setting "
                "reuse_actors=True to reduce actor creation "
                "overheads.".format(setup_time)
            )
        log_sys_usage = self.config.get("log_sys_usage", False)
        self._start_time = start_time
        self._warmup_time = None
        self._monitor = UtilMonitor(start=log_sys_usage)

        self.remote_checkpoint_dir = None
        self.custom_syncer = None
        self.sync_timeout = None
        self.sync_num_retries = int(os.getenv("TUNE_CHECKPOINT_CLOUD_RETRY_NUM", "3"))
        self.sync_sleep_time = float(
            os.getenv("TUNE_CHECKPOINT_CLOUD_RETRY_WAIT_TIME_S", "1")
        )

        # Check, whether `training_iteration` is still a tune.Trainable property
        # and has not been overridden by the user in the attempt to implement the
        # algos logic (this should be done now inside `training_step`).
        try:
            assert isinstance(self.training_iteration, int)
        except AssertionError:
            raise AssertionError(
                "Your Algorithm's `training_iteration` seems to be overridden by your "
                "custom training logic! To solve this problem, simply rename your "
                "`self.training_iteration()` method into `self.training_step`."
            )

    @OverrideToImplementCustomLogic_CallToSuperRecommended
    @override(Algorithm)
    def setup(self, config: AlgorithmConfig) -> None:

        # Setup our config: Merge the user-supplied config dict (which could
        # be a partial config dict) with the class' default.
        if not isinstance(config, AlgorithmConfig):
            assert isinstance(config, PartialAlgorithmConfigDict)
            config_obj = self.get_default_config()
            if not isinstance(config_obj, AlgorithmConfig):
                assert isinstance(config, PartialAlgorithmConfigDict)
                config_obj = AlgorithmConfig().from_dict(config_obj)
            config_obj.update_from_dict(config)
            config_obj.env = self._env_id
            self.config = config_obj

        # Set Algorithm's seed after we have - if necessary - enabled
        # tf eager-execution.
        update_global_seed_if_necessary(self.config.framework_str, self.config.seed)

        self._record_usage(self.config)

        self.callbacks = self.config["callbacks"]()

        # Create local replay buffer if necessary.
        self.local_replay_buffer = self._create_local_replay_buffer_if_necessary(
            self.config
        )

        # Create a dict, mapping ActorHandles to sets of open remote
        # requests (object refs). This way, we keep track, of which actors
        # inside this Algorithm (e.g. a remote RolloutWorker) have
        # already been sent how many (e.g. `sample()`) requests.
        self.remote_requests_in_flight: DefaultDict[
            ActorHandle, Set[ray.ObjectRef]
        ] = defaultdict(set)

        self.workers: Optional[WorkerSet] = None
        self.train_exec_impl = None

        # Offline RL settings.
        input_evaluation = self.config.get("input_evaluation")
        if input_evaluation is not None and input_evaluation is not DEPRECATED_VALUE:
            ope_dict = {str(ope): {"type": ope} for ope in input_evaluation}
            deprecation_warning(
                old="config.input_evaluation={}".format(input_evaluation),
                new="config.evaluation(evaluation_config=config.overrides("
                f"off_policy_estimation_methods={ope_dict}"
                "))",
                error=True,
                help="Running OPE during training is not recommended.",
            )
            self.config["off_policy_estimation_methods"] = ope_dict

        # Deprecated way of implementing Trainer sub-classes (or "templates"
        # via the `build_trainer` utility function).
        # Instead, sub-classes should override the Trainable's `setup()`
        # method and call super().setup() from within that override at some
        # point.
        # Old design: Override `Trainer._init`.
        _init = False
        try:
            self._init(self.config, self.env_creator)
            _init = True
        # New design: Override `Trainable.setup()` (as indented by tune.Trainable)
        # and do or don't call `super().setup()` from within your override.
        # By default, `super().setup()` will create both worker sets:
        # "rollout workers" for collecting samples for training and - if
        # applicable - "evaluation workers" for evaluation runs in between or
        # parallel to training.
        # TODO: Deprecate `_init()` and remove this try/except block.
        except NotImplementedError:
            pass

        # Only if user did not override `_init()`:
        if _init is False:
            # - Create rollout workers here automatically.
            # - Run the execution plan to create the local iterator to `next()`
            #   in each training iteration.
            # This matches the behavior of using `build_trainer()`, which
            # has been deprecated.
            self.workers = WorkerSet(
                env_creator=self.env_creator,
                validate_env=self.validate_env,
                default_policy_class=self.get_default_policy_class(self.config),
                config=self.config,
                num_workers=self.config["num_workers"],
                local_worker=True,
                logdir=self.logdir,
            )

            # TODO (avnishn): Remove the execution plan API by q1 2023
            # Function defining one single training iteration's behavior.
            if self.config["_disable_execution_plan_api"]:
                # Ensure remote workers are initially in sync with the local worker.
                self.workers.sync_weights()
            # LocalIterator-creating "execution plan".
            # Only call this once here to create `self.train_exec_impl`,
            # which is a ray.util.iter.LocalIterator that will be `next`'d
            # on each training iteration.
            else:
                self.train_exec_impl = self.execution_plan(
                    self.workers, self.config, **self._kwargs_for_execution_plan()
                )

            # Now that workers have been created, update our policies
            # dict in config[multiagent] (with the correct original/
            # unpreprocessed spaces).
            self.config["multiagent"][
                "policies"
            ] = self.workers.local_worker().policy_dict

        # Compile, validate, and freeze an evaluation config.
        self.evaluation_config = self.config.get_evaluation_config_object()
        self.evaluation_config.validate()
        self.evaluation_config.freeze()

        # Evaluation WorkerSet setup.
        # User would like to setup a separate evaluation worker set.
        # Note: We skip workerset creation if we need to do offline evaluation
        if self._should_create_evaluation_rollout_workers(self.evaluation_config):
            for evaluation_env in self.config["eval_envs"]:
                _, env_creator = self._get_env_id_and_creator(
                    evaluation_env, self.evaluation_config
                )

                # Create a separate evaluation worker set for evaluation.
                # If evaluation_num_workers=0, use the evaluation set's local
                # worker for evaluation, otherwise, use its remote workers
                # (parallelized evaluation).
                self.evaluation_workers.append(
                    WorkerSet(
                        env_creator=env_creator,
                        validate_env=None,
                        default_policy_class=self.get_default_policy_class(self.config),
                        config=self.evaluation_config,
                        num_workers=self.config["evaluation_num_workers"],
                        # Don't even create a local worker if num_workers > 0.
                        local_worker=False,
                        logdir=self.logdir,
                    )
                )

            if self.config["enable_async_evaluation"]:
                self._evaluation_weights_seq_number = 0

        self.evaluation_dataset = None
        if (
            self.evaluation_config.off_policy_estimation_methods
            and not self.evaluation_config.ope_split_batch_by_episode
        ):
            # the num worker is set to 0 to avoid creating shards. The dataset will not
            # be repartioned to num_workers blocks.
            logger.info("Creating evaluation dataset ...")
            ds, _ = get_dataset_and_shards(self.evaluation_config, num_workers=0)

            # Dataset should be in form of one episode per row. in case of bandits each
            # row is just one time step. To make the computation more efficient later
            # we remove the time dimension here.
            parallelism = self.evaluation_config.evaluation_num_workers or 1
            batch_size = max(ds.count() // parallelism, 1)
            self.evaluation_dataset = ds.map_batches(
                remove_time_dim, batch_size=batch_size
            )
            logger.info("Evaluation dataset created")

        self.reward_estimators: Dict[str, OffPolicyEstimator] = {}
        ope_types = {
            "is": ImportanceSampling,
            "wis": WeightedImportanceSampling,
            "dm": DirectMethod,
            "dr": DoublyRobust,
        }
        for name, method_config in self.config["off_policy_estimation_methods"].items():
            method_type = method_config.pop("type")
            if method_type in ope_types:
                deprecation_warning(
                    old=method_type,
                    new=str(ope_types[method_type]),
                    error=True,
                )
                method_type = ope_types[method_type]
            elif isinstance(method_type, str):
                logger.log(0, "Trying to import from string: " + method_type)
                mod, obj = method_type.rsplit(".", 1)
                mod = importlib.import_module(mod)
                method_type = getattr(mod, obj)
            if isinstance(method_type, type) and issubclass(
                method_type, OfflineEvaluator
            ):
                # TODO(kourosh) : Add an integration test for all these
                # offline evaluators.
                policy = self.get_policy()
                if issubclass(method_type, OffPolicyEstimator):
                    method_config["gamma"] = self.config["gamma"]
                self.reward_estimators[name] = method_type(policy, **method_config)
            else:
                raise ValueError(
                    f"Unknown off_policy_estimation type: {method_type}! Must be "
                    "either a class path or a sub-class of ray.rllib."
                    "offline.offline_evaluator::OfflineEvaluator"
                )

        # Run `on_algorithm_init` callback after initialization is done.
        self.callbacks.on_algorithm_init(algorithm=self)

    @override(Algorithm)
    def step(self, CL_option=None, lock_update=False, memory_generator=None) -> ResultDict:
        """Implements the main `Trainer.train()` logic.

        Takes n attempts to perform a single training step. Thereby
        catches RayErrors resulting from worker failures. After n attempts,
        fails gracefully.

        Override this method in your Trainer sub-classes if you would like to
        handle worker failures yourself.
        Otherwise, override only `training_step()` to implement the core
        algorithm logic.

        Returns:
            The results dict with stats/infos on sampling, training,
            and - if required - evaluation.
        """
        # Do we have to run `self.evaluate()` this iteration?
        # `self.iteration` gets incremented after this function returns,
        # meaning that e. g. the first time this function is called,
        # self.iteration will be 0.
        evaluate_this_iter = (
            self.config.evaluation_interval is not None
            and (self.iteration + 1) % self.config.evaluation_interval == 0
        )

        # Results dict for training (and if appolicable: evaluation).
        results: ResultDict = {}

        # Parallel eval + training: Kick off evaluation-loop and parallel train() call.
        if evaluate_this_iter and self.config["evaluation_parallel_to_training"] and not lock_update:
            (
                results,
                train_iter_ctx,
            ) = self._run_one_training_iteration_and_evaluation_in_parallel()
        # - No evaluation necessary, just run the next training iteration.
        # - We have to evaluate in this training iteration, but no parallelism ->
        #   evaluate after the training iteration is entirely done.
        else:
            results, train_iter_ctx = self._run_one_training_iteration(CL_option, lock_update, memory_generator)

        # Sequential: Train (already done above), then evaluate.
        if evaluate_this_iter and not self.config["evaluation_parallel_to_training"] and not lock_update:
            results.update(self._run_one_evaluation(train_future=None))

        # Attach latest available evaluation results to train results,
        # if necessary.
        if not evaluate_this_iter and self.config["always_attach_evaluation_results"] and not lock_update:
            assert isinstance(
                self.evaluation_metrics, dict
            ), "Trainer.evaluate() needs to return a dict."
            results.update(self.evaluation_metrics)

        if hasattr(self, "workers") and isinstance(self.workers, WorkerSet):
            # Sync filters on workers.
            self._sync_filters_if_needed(
                from_worker=self.workers.local_worker(),
                workers=self.workers,
                timeout_seconds=self.config[
                    "sync_filters_on_rollout_workers_timeout_s"
                ],
            )
            # TODO (avnishn): Remove the execution plan API by q1 2023
            # Collect worker metrics and add combine them with `results`.
            if self.config["_disable_execution_plan_api"]:
                episodes_this_iter = collect_episodes(
                    self.workers,
                    self._remote_worker_ids_for_metrics(),
                    timeout_seconds=self.config["metrics_episode_collection_timeout_s"],
                )
                results = self._compile_iteration_results(
                    episodes_this_iter=episodes_this_iter,
                    step_ctx=train_iter_ctx,
                    iteration_results=results,
                )

        # Check `env_task_fn` for possible update of the env's task.
        if self.config["env_task_fn"] is not None:
            if not callable(self.config["env_task_fn"]):
                raise ValueError(
                    "`env_task_fn` must be None or a callable taking "
                    "[train_results, env, env_ctx] as args!"
                )

            def fn(env, env_context, task_fn):
                new_task = task_fn(results, env, env_context)
                cur_task = env.get_task()
                if cur_task != new_task:
                    env.set_task(new_task)

            fn = functools.partial(fn, task_fn=self.config["env_task_fn"])
            self.workers.foreach_env_with_context(fn)

        return results

    @PublicAPI
    def evaluate(
        self,
        ewi,
        duration_fn: Optional[Callable[[int], int]] = None,
    ) -> dict:
        """Evaluates current policy under `evaluation_config` settings.

        Note that this default implementation does not do anything beyond
        merging evaluation_config with the normal trainer config.

        Args:
            duration_fn: An optional callable taking the already run
                num episodes as only arg and returning the number of
                episodes left to run. It's used to find out whether
                evaluation should continue.
        """
        # Call the `_before_evaluate` hook.
        self._before_evaluate(ewi)

        if self.evaluation_dataset is not None:
            return {"evaluation"+str(ewi): self._run_offline_evaluation()}

        # Sync weights to the evaluation WorkerSet.
        self.evaluation_workers[ewi].sync_weights(
            from_worker=self.workers.local_worker()
        )
        self._sync_filters_if_needed(
            from_worker=self.workers.local_worker(),
            workers=self.evaluation_workers[ewi],
            timeout_seconds=self.config[
                "sync_filters_on_rollout_workers_timeout_s"
            ],
        )

        self.callbacks.on_evaluate_start(algorithm=self)

        if self.config["custom_eval_function"]:
            logger.info(
                "Running custom eval function {}".format(
                    self.config["custom_eval_function"]
                )
            )
            metrics = self.config["custom_eval_function"](self, self.evaluation_workers[ewi])
            if not metrics or not isinstance(metrics, dict):
                raise ValueError(
                    "Custom eval function must return "
                    "dict of metrics, got {}.".format(metrics)
                )
        else:
            # How many episodes/timesteps do we need to run?
            # In "auto" mode (only for parallel eval + training): Run as long
            # as training lasts.
            unit = self.config["evaluation_duration_unit"]
            eval_cfg = self.evaluation_config
            rollout = eval_cfg["rollout_fragment_length"]
            num_envs = eval_cfg["num_envs_per_worker"]
            auto = self.config["evaluation_duration"] == "auto"
            duration = (
                self.config["evaluation_duration"]
                if not auto
                else (self.config["evaluation_num_workers"] or 1)
                * (1 if unit == "episodes" else rollout)
            )
            agent_steps_this_iter = 0
            env_steps_this_iter = 0

            # Default done-function returns True, whenever num episodes
            # have been completed.
            if duration_fn is None:

                def duration_fn(num_units_done):
                    return duration - num_units_done

            logger.info(f"Evaluating current policy for {duration} {unit}.")

            metrics = None
            all_batches = []
            # No evaluation worker set ->
            # Do evaluation using the local worker. Expect error due to the
            # local worker not having an env.
            if self.evaluation_workers[ewi].num_remote_workers() == 0:
                # If unit=episodes -> Run n times `sample()` (each sample
                # produces exactly 1 episode).
                # If unit=ts -> Run 1 `sample()` b/c the
                # `rollout_fragment_length` is exactly the desired ts.
                iters = duration if unit == "episodes" else 1
                for _ in range(iters):
                    batch = self.evaluation_workers[ewi].local_worker().sample()
                    agent_steps_this_iter += batch.agent_steps()
                    env_steps_this_iter += batch.env_steps()
                    if self.reward_estimators:
                        all_batches.append(batch)

            # Evaluation worker set has n remote workers.
            elif self.evaluation_workers[ewi].num_healthy_remote_workers() > 0:
                # How many episodes have we run (across all eval workers)?
                num_units_done = 0
                _round = 0
                # In case all of the remote evaluation workers die during a round
                # of evaluation, we need to stop.
                while True and self.evaluation_workers[ewi].num_healthy_remote_workers() > 0:
                    units_left_to_do = duration_fn(num_units_done)
                    if units_left_to_do <= 0:
                        break

                    _round += 1
                    unit_per_remote_worker = (
                        1 if unit == "episodes" else rollout * num_envs
                    )
                    # Select proper number of evaluation workers for this round.
                    selected_eval_worker_ids = [
                        worker_id
                        for i, worker_id in enumerate(
                            self.evaluation_workers[ewi].healthy_worker_ids()
                        )
                        if i * unit_per_remote_worker < units_left_to_do
                    ]
                    batches = self.evaluation_workers[ewi].foreach_worker(
                        func=lambda w: w.sample(),
                        local_worker=False,
                        remote_worker_ids=selected_eval_worker_ids,
                        timeout_seconds=self.config["evaluation_sample_timeout_s"],
                    )
                    if len(batches) != len(selected_eval_worker_ids):
                        logger.warning(
                            "Calling `sample()` on your remote evaluation worker(s) "
                            "resulted in a timeout (after the configured "
                            f"{self.config['evaluation_sample_timeout_s']} seconds)! "
                            "Try to set `evaluation_sample_timeout_s` in your config"
                            " to a larger value."
                            + (
                                " If your episodes don't terminate easily, you may "
                                "also want to set `evaluation_duration_unit` to "
                                "'timesteps' (instead of 'episodes')."
                                if unit == "episodes"
                                else ""
                            )
                        )
                        break

                    _agent_steps = sum(b.agent_steps() for b in batches)
                    _env_steps = sum(b.env_steps() for b in batches)
                    # 1 episode per returned batch.
                    if unit == "episodes":
                        num_units_done += len(batches)
                        # Make sure all batches are exactly one episode.
                        for ma_batch in batches:
                            ma_batch = ma_batch.as_multi_agent()
                            for batch in ma_batch.policy_batches.values():
                                assert batch.is_terminated_or_truncated()
                    # n timesteps per returned batch.
                    else:
                        num_units_done += (
                            _agent_steps
                            if self.config.count_steps_by == "agent_steps"
                            else _env_steps
                        )
                    if self.reward_estimators:
                        # TODO: (kourosh) This approach will cause an OOM issue when
                        # the dataset gets huge (should be ok for now).
                        all_batches.extend(batches)

                    agent_steps_this_iter += _agent_steps
                    env_steps_this_iter += _env_steps

                    logger.info(
                        f"Ran round {_round} of parallel evaluation "
                        f"({num_units_done}/{duration if not auto else '?'} "
                        f"{unit} done)"
                    )
            else:
                # Can't find a good way to run this evaluation.
                # Wait for next iteration.
                pass

            if metrics is None:
                metrics = collect_metrics(
                    self.evaluation_workers[ewi],
                    keep_custom_metrics=self.config["keep_per_episode_custom_metrics"],
                    timeout_seconds=eval_cfg["metrics_episode_collection_timeout_s"],
                )
            metrics[NUM_AGENT_STEPS_SAMPLED_THIS_ITER] = agent_steps_this_iter
            metrics[NUM_ENV_STEPS_SAMPLED_THIS_ITER] = env_steps_this_iter
            # TODO: Remove this key at some point. Here for backward compatibility.
            metrics["timesteps_this_iter"] = env_steps_this_iter

            # Compute off-policy estimates
            estimates = defaultdict(list)
            # for each batch run the estimator's fwd pass
            for name, estimator in self.reward_estimators.items():
                for batch in all_batches:
                    estimate_result = estimator.estimate(
                        batch,
                        split_batch_by_episode=self.config[
                            "ope_split_batch_by_episode"
                        ],
                    )
                    estimates[name].append(estimate_result)

            # collate estimates from all batches
            if estimates:
                metrics["off_policy_estimator"] = {}
                for name, estimate_list in estimates.items():
                    avg_estimate = tree.map_structure(
                        lambda *x: np.mean(x, axis=0), *estimate_list
                    )
                    metrics["off_policy_estimator"][name] = avg_estimate

        # Evaluation does not run for every step.
        # Save evaluation metrics on trainer, so it can be attached to
        # subsequent step results as latest evaluation result.
        self.evaluation_metrics = {"evaluation"+str(ewi): metrics}

        # Trigger `on_evaluate_end` callback.
        self.callbacks.on_evaluate_end(
            algorithm=self, evaluation_metrics=self.evaluation_metrics
        )

        # Also return the results here for convenience.
        return self.evaluation_metrics

    @ExperimentalAPI
    def _evaluate_async(
        self,
        ewi,
        duration_fn: Optional[Callable[[int], int]] = None,
    ) -> dict:
        """Evaluates current policy under `evaluation_config` settings.

        Uses the AsyncParallelRequests manager to send frequent `sample.remote()`
        requests to the evaluation RolloutWorkers and collect the results of these
        calls. Handles worker failures (or slowdowns) gracefully due to the asynch'ness
        and the fact that other eval RolloutWorkers can thus cover the workload.

        Important Note: This will replace the current `self.evaluate()` method as the
        default in the future.

        Args:
            duration_fn: An optional callable taking the already run
                num episodes as only arg and returning the number of
                episodes left to run. It's used to find out whether
                evaluation should continue.
        """
        # How many episodes/timesteps do we need to run?
        # In "auto" mode (only for parallel eval + training): Run as long
        # as training lasts.
        unit = self.config["evaluation_duration_unit"]
        eval_cfg = self.evaluation_config
        rollout = eval_cfg["rollout_fragment_length"]
        num_envs = eval_cfg["num_envs_per_worker"]
        auto = self.config["evaluation_duration"] == "auto"
        duration = (
            self.config["evaluation_duration"]
            if not auto
            else (self.config["evaluation_num_workers"] or 1)
            * (1 if unit == "episodes" else rollout)
        )

        # Call the `_before_evaluate` hook.
        self._before_evaluate()

        # TODO(Jun): Implement solution via connectors.
        self._sync_filters_if_needed(
            from_worker=self.workers.local_worker(),
            workers=self.evaluation_workers[ewi],
            timeout_seconds=eval_cfg.get("sync_filters_on_rollout_workers_timeout_s"),
        )

        if self.config["custom_eval_function"]:
            raise ValueError(
                "`custom_eval_function` not supported in combination "
                "with `enable_async_evaluation=True` config setting!"
            )

        agent_steps_this_iter = 0
        env_steps_this_iter = 0

        logger.info(f"Evaluating current policy for {duration} {unit}.")

        all_batches = []

        # Default done-function returns True, whenever num episodes
        # have been completed.
        if duration_fn is None:

            def duration_fn(num_units_done):
                return duration - num_units_done

        # Put weights only once into object store and use same object
        # ref to synch to all workers.
        self._evaluation_weights_seq_number += 1
        weights_ref = ray.put(self.workers.local_worker().get_weights())
        weights_seq_no = self._evaluation_weights_seq_number

        def remote_fn(worker):
            # Pass in seq-no so that eval workers may ignore this call if no update has
            # happened since the last call to `remote_fn` (sample).
            worker.set_weights(
                weights=ray.get(weights_ref), weights_seq_no=weights_seq_no
            )
            batch = worker.sample()
            metrics = worker.get_metrics()
            return batch, metrics, weights_seq_no

        rollout_metrics = []

        # How many episodes have we run (across all eval workers)?
        num_units_done = 0
        _round = 0

        while self.evaluation_workers[ewi].num_healthy_remote_workers() > 0:
            units_left_to_do = duration_fn(num_units_done)
            if units_left_to_do <= 0:
                break

            _round += 1
            # Get ready evaluation results and metrics asynchronously.
            self.evaluation_workers[ewi].foreach_worker_async(
                func=remote_fn,
                healthy_only=True,
            )
            eval_results = self.evaluation_workers[ewi].fetch_ready_async_reqs()

            batches = []
            i = 0
            for _, result in eval_results:
                batch, metrics, seq_no = result
                # Ignore results, if the weights seq-number does not match (is
                # from a previous evaluation step) OR if we have already reached
                # the configured duration (e.g. number of episodes to evaluate
                # for).
                if seq_no == self._evaluation_weights_seq_number and (
                    i * (1 if unit == "episodes" else rollout * num_envs)
                    < units_left_to_do
                ):
                    batches.append(batch)
                    rollout_metrics.extend(metrics)
                i += 1

            _agent_steps = sum(b.agent_steps() for b in batches)
            _env_steps = sum(b.env_steps() for b in batches)

            # 1 episode per returned batch.
            if unit == "episodes":
                num_units_done += len(batches)
                # Make sure all batches are exactly one episode.
                for ma_batch in batches:
                    ma_batch = ma_batch.as_multi_agent()
                    for batch in ma_batch.policy_batches.values():
                        assert batch.is_terminated_or_truncated()
            # n timesteps per returned batch.
            else:
                num_units_done += (
                    _agent_steps
                    if self.config.count_steps_by == "agent_steps"
                    else _env_steps
                )

            if self.reward_estimators:
                all_batches.extend(batches)

            agent_steps_this_iter += _agent_steps
            env_steps_this_iter += _env_steps

            logger.info(
                f"Ran round {_round} of parallel evaluation "
                f"({num_units_done}/{duration if not auto else '?'} "
                f"{unit} done)"
            )

        metrics = summarize_episodes(
            rollout_metrics,
            keep_custom_metrics=eval_cfg["keep_per_episode_custom_metrics"],
        )

        metrics[NUM_AGENT_STEPS_SAMPLED_THIS_ITER] = agent_steps_this_iter
        metrics[NUM_ENV_STEPS_SAMPLED_THIS_ITER] = env_steps_this_iter
        # TODO: Remove this key at some point. Here for backward compatibility.
        metrics["timesteps_this_iter"] = env_steps_this_iter

        if self.reward_estimators:
            # Compute off-policy estimates
            metrics["off_policy_estimator"] = {}
            total_batch = concat_samples(all_batches)
            for name, estimator in self.reward_estimators.items():
                estimates = estimator.estimate(total_batch)
                metrics["off_policy_estimator"][name] = estimates

        # Evaluation does not run for every step.
        # Save evaluation metrics on trainer, so it can be attached to
        # subsequent step results as latest evaluation result.
        self.evaluation_metrics = {"evaluation"+str(ewi): metrics}

        # Trigger `on_evaluate_end` callback.
        self.callbacks.on_evaluate_end(
            algorithm=self, evaluation_metrics=self.evaluation_metrics
        )

        # Return evaluation results.
        return self.evaluation_metrics

    @PublicAPI
    def add_policy(
        self,
        policy_id: PolicyID,
        policy_cls: Optional[Type[Policy]] = None,
        policy: Optional[Policy] = None,
        *,
        observation_space: Optional[gym.spaces.Space] = None,
        action_space: Optional[gym.spaces.Space] = None,
        config: Optional[Union[AlgorithmConfig, PartialAlgorithmConfigDict]] = None,
        policy_state: Optional[PolicyState] = None,
        policy_mapping_fn: Optional[Callable[[AgentID, EpisodeID], PolicyID]] = None,
        policies_to_train: Optional[
            Union[
                Container[PolicyID],
                Callable[[PolicyID, Optional[SampleBatchType]], bool],
            ]
        ] = None,
        evaluation_workers: bool = True,
        # Deprecated.
        workers: Optional[List[Union[RolloutWorker, ActorHandle]]] = DEPRECATED_VALUE,
    ) -> Optional[Policy]:
        """Adds a new policy to this Algorithm.

        Args:
            policy_id: ID of the policy to add.
                IMPORTANT: Must not contain characters that
                are also not allowed in Unix/Win filesystems, such as: `<>:"/|?*`,
                or a dot, space or backslash at the end of the ID.
            policy_cls: The Policy class to use for constructing the new Policy.
                Note: Only one of `policy_cls` or `policy` must be provided.
            policy: The Policy instance to add to this algorithm. If not None, the
                given Policy object will be directly inserted into the Algorithm's
                local worker and clones of that Policy will be created on all remote
                workers as well as all evaluation workers.
                Note: Only one of `policy_cls` or `policy` must be provided.
            observation_space: The observation space of the policy to add.
                If None, try to infer this space from the environment.
            action_space: The action space of the policy to add.
                If None, try to infer this space from the environment.
            config: The config object or overrides for the policy to add.
            policy_state: Optional state dict to apply to the new
                policy instance, right after its construction.
            policy_mapping_fn: An optional (updated) policy mapping function
                to use from here on. Note that already ongoing episodes will
                not change their mapping but will use the old mapping till
                the end of the episode.
            policies_to_train: An optional list of policy IDs to be trained
                or a callable taking PolicyID and SampleBatchType and
                returning a bool (trainable or not?).
                If None, will keep the existing setup in place. Policies,
                whose IDs are not in the list (or for which the callable
                returns False) will not be updated.
            evaluation_workers: Whether to add the new policy also
                to the evaluation WorkerSet.
            workers: A list of RolloutWorker/ActorHandles (remote
                RolloutWorkers) to add this policy to. If defined, will only
                add the given policy to these workers.

        Returns:
            The newly added policy (the copy that got added to the local
            worker). If `workers` was provided, None is returned.
        """
        validate_policy_id(policy_id, error=True)

        if workers is not DEPRECATED_VALUE:
            deprecation_warning(
                old="workers",
                help=(
                    "The `workers` argument to `Algorithm.add_policy()` is deprecated "
                    "and no-op now. Please do not use it anymore."
                ),
                error=False,
            )

        self.workers.add_policy(
            policy_id,
            policy_cls,
            policy,
            observation_space=observation_space,
            action_space=action_space,
            config=config,
            policy_state=policy_state,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
        )

        # Add to evaluation workers, if necessary.
        if evaluation_workers is True:
            for ewi in range(len(self.evaluation_workers)):
                self.evaluation_workers[ewi].add_policy(
                    policy_id,
                    policy_cls,
                    policy,
                    observation_space=observation_space,
                    action_space=action_space,
                    config=config,
                    policy_state=policy_state,
                    policy_mapping_fn=policy_mapping_fn,
                    policies_to_train=policies_to_train,
                )

        # Return newly added policy (from the local rollout worker).
        return self.get_policy(policy_id)

    @PublicAPI
    def remove_policy(
        self,
        policy_id: PolicyID = DEFAULT_POLICY_ID,
        *,
        policy_mapping_fn: Optional[Callable[[AgentID], PolicyID]] = None,
        policies_to_train: Optional[
            Union[
                Container[PolicyID],
                Callable[[PolicyID, Optional[SampleBatchType]], bool],
            ]
        ] = None,
        evaluation_workers: bool = True,
    ) -> None:
        """Removes a new policy from this Algorithm.

        Args:
            policy_id: ID of the policy to be removed.
            policy_mapping_fn: An optional (updated) policy mapping function
                to use from here on. Note that already ongoing episodes will
                not change their mapping but will use the old mapping till
                the end of the episode.
            policies_to_train: An optional list of policy IDs to be trained
                or a callable taking PolicyID and SampleBatchType and
                returning a bool (trainable or not?).
                If None, will keep the existing setup in place. Policies,
                whose IDs are not in the list (or for which the callable
                returns False) will not be updated.
            evaluation_workers: Whether to also remove the policy from the
                evaluation WorkerSet.
        """

        def fn(worker):
            worker.remove_policy(
                policy_id=policy_id,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=policies_to_train,
            )

        self.workers.foreach_worker(fn, local_worker=True, healthy_only=True)
        if evaluation_workers:
            for ewi in range(len(self.evaluation_workers)):
                self.evaluation_workers[ewi].foreach_worker(
                    fn,
                    local_worker=True,
                    healthy_only=True,
                )

    @override(Trainable)
    def cleanup(self) -> None:
        # Stop all workers.
        if hasattr(self, "workers") and self.workers is not None:
            self.workers.stop()
        for ewi in range(len(self.evaluation_workers)):
            self.evaluation_workers[ewi].stop()

    @DeveloperAPI
    def _before_evaluate(self, ewi):
        """Pre-evaluation callback."""
        pass

    @staticmethod
    @ExperimentalAPI
    def validate_env(env: EnvType, env_context: EnvContext) -> None:
        """Env validator function for this Algorithm class.

        Override this in child classes to define custom validation
        behavior.

        Args:
            env: The (sub-)environment to validate. This is normally a
                single sub-environment (e.g. a gym.Env) within a vectorized
                setup.
            env_context: The EnvContext to configure the environment.

        Raises:
            Exception in case something is wrong with the given environment.
        """
        pass

    @PublicAPI
    def __setstate__(self, state) -> None:
        """Sets the algorithm to the provided state.

        Args:
            state: The state dict to restore this Algorithm instance to. `state` may
                have been returned by a call to an Algorithm's `__getstate__()` method.
        """
        # TODO (sven): Validate that our config and the config in state are compatible.
        #  For example, the model architectures may differ.
        #  Also, what should the behavior be if e.g. some training parameter
        #  (e.g. lr) changed?

        if hasattr(self, "workers") and "worker" in state:
            self.workers.local_worker().set_state(state["worker"])
            remote_state = ray.put(state["worker"])
            self.workers.foreach_worker(
                lambda w: w.set_state(ray.get(remote_state)),
                local_worker=False,
                healthy_only=False,
            )
            # If evaluation workers are used, also restore the policies
            # there in case they are used for evaluation purpose.
            for ewi in range(len(self.evaluation_workers)):
                self.evaluation_workers[ewi].foreach_worker(
                    lambda w: w.set_state(ray.get(remote_state)),
                    local_worker=False,
                    healthy_only=False,
                )
        # If necessary, restore replay data as well.
        if self.local_replay_buffer is not None:
            # TODO: Experimental functionality: Restore contents of replay
            #  buffer from checkpoint, only if user has configured this.
            if self.config.get("store_buffer_in_checkpoints"):
                if "local_replay_buffer" in state:
                    self.local_replay_buffer.set_state(state["local_replay_buffer"])
                else:
                    logger.warning(
                        "`store_buffer_in_checkpoints` is True, but no replay "
                        "data found in state!"
                    )
            elif "local_replay_buffer" in state and log_once(
                "no_store_buffer_in_checkpoints_but_data_found"
            ):
                logger.warning(
                    "`store_buffer_in_checkpoints` is False, but some replay "
                    "data found in state!"
                )

        if self.train_exec_impl is not None:
            self.train_exec_impl.shared_metrics.get().restore(state["train_exec_impl"])
        elif "counters" in state:
            self._counters = state["counters"]

    def _run_one_training_iteration(self, CL_option=None, lock_update=False, memory_generator=None) -> Tuple[ResultDict, "TrainIterCtx"]:
        """Runs one training iteration (self.iteration will be +1 after this).

        Calls `self.training_step()` repeatedly until the minimum time (sec),
        sample- or training steps have been reached.

        Returns:
            The results dict from the training iteration.
        """
        # In case we are training (in a thread) parallel to evaluation,
        # we may have to re-enable eager mode here (gets disabled in the
        # thread).
        if self.config.get("framework") == "tf2" and not tf.executing_eagerly():
            tf1.enable_eager_execution()

        results = None
        # Create a step context ...
        with TrainIterCtx(algo=self) as train_iter_ctx:
            # .. so we can query it whether we should stop the iteration loop (e.g.
            # when we have reached `min_time_s_per_iteration`).
            while not train_iter_ctx.should_stop(results):
                # Try to train one step.
                # TODO (avnishn): Remove the execution plan API by q1 2023
                with self._timers[TRAINING_ITERATION_TIMER]:
                    if self.config._disable_execution_plan_api:
                        results = self.training_step(CL_option, lock_update, memory_generator)
                    else:
                        results = next(self.train_exec_impl)

        # With training step done. Try to bring failed workers back.
        self.restore_workers(self.workers)

        return results, train_iter_ctx

    def _run_one_evaluation_(
        self,
        ewi,
        train_future: Optional[concurrent.futures.ThreadPoolExecutor] = None,
    ) -> ResultDict:
        """Runs evaluation step via `self.evaluate()` and handling worker failures.

        Args:
            train_future: In case, we are training and avaluating in parallel,
                this arg carries the currently running ThreadPoolExecutor
                object that runs the training iteration

        Returns:
            The results dict from the evaluation call.
        """

        eval_results = {
            "evaluation"+str(ewi): {
                "episode_reward_max": np.nan,
                "episode_reward_min": np.nan,
                "episode_reward_mean": np.nan,
            }
        }

        eval_func_to_use = (
            self._evaluate_async
            if self.config.enable_async_evaluation
            else self.evaluate
        )

        if self.config.evaluation_duration == "auto":
            assert (
                train_future is not None and self.config.evaluation_parallel_to_training
            )
            unit = self.config.evaluation_duration_unit
            eval_results = eval_func_to_use(
                ewi,
                duration_fn=functools.partial(
                    self._automatic_evaluation_duration_fn,
                    unit,
                    self.config.evaluation_num_workers,
                    self.evaluation_config,
                    train_future,
                )
            )
        # Run `self.evaluate()` only once per training iteration.
        else:
            eval_results = eval_func_to_use(ewi)

        # After evaluation, do a round of health check to see if any of
        # the failed workers are back.
        self.restore_workers(self.evaluation_workers[ewi])

        # Add number of healthy evaluation workers after this iteration.
        eval_results["evaluation"+str(ewi)][
            "num_healthy_workers"
        ] = self.evaluation_workers[ewi].num_healthy_remote_workers()
        eval_results["evaluation"+str(ewi)][
            "num_in_flight_async_reqs"
        ] = self.evaluation_workers[ewi].num_in_flight_async_reqs()
        eval_results["evaluation"+str(ewi)][
            "num_remote_worker_restarts"
        ] = self.evaluation_workers[ewi].num_remote_worker_restarts()

        return eval_results

    def _run_one_evaluation(
        self,
        train_future: Optional[concurrent.futures.ThreadPoolExecutor] = None,
    ) -> ResultDict:
        """Runs evaluation step via `self.evaluate()` and handling worker failures.

        Args:
            train_future: In case, we are training and avaluating in parallel,
                this arg carries the currently running ThreadPoolExecutor
                object that runs the training iteration

        Returns:
            The results dict from the evaluation call.
        """
        eval_results = {}
        for ewi in range(len(self.evaluation_workers)):
            eval_results["evaluation"+str(ewi)] = {
                "episode_reward_max": np.nan,
                "episode_reward_min": np.nan,
                "episode_reward_mean": np.nan,
            }
            
            eval_result = self._run_one_evaluation_(ewi=ewi, train_future=train_future)
            eval_results.update(eval_result)

        return eval_results

    def restore(
        self,
        checkpoint_path: Union[str, Checkpoint],
        checkpoint_node_ip: Optional[str] = None,
        fallback_to_latest: bool = False,
    ):
        """Restores training state from a given model checkpoint.

        These checkpoints are returned from calls to save().

        Subclasses should override ``load_checkpoint()`` instead to
        restore state.
        This method restores additional metadata saved with the checkpoint.

        `checkpoint_path` should match with the return from ``save()``.

        `checkpoint_path` can be
        `~/ray_results/exp/MyTrainable_abc/
        checkpoint_00000/checkpoint`. Or,
        `~/ray_results/exp/MyTrainable_abc/checkpoint_00000`.

        `self.logdir` should generally be corresponding to `checkpoint_path`,
        for example, `~/ray_results/exp/MyTrainable_abc`.

        `self.remote_checkpoint_dir` in this case, is something like,
        `REMOTE_CHECKPOINT_BUCKET/exp/MyTrainable_abc`

        Args:
            checkpoint_path: Path to restore checkpoint from. If this
                path does not exist on the local node, it will be fetched
                from external (cloud) storage if available, or restored
                from a remote node.
            checkpoint_node_ip: If given, try to restore
                checkpoint from this node if it doesn't exist locally or
                on cloud storage.
            fallback_to_latest: If True, will try to recover the
                latest available checkpoint if the given ``checkpoint_path``
                could not be found.

        """
        # Ensure Checkpoints are converted
        if isinstance(checkpoint_path, Checkpoint):
            return self._restore_from_checkpoint_obj(checkpoint_path)

        if not self._maybe_load_from_cloud(checkpoint_path) and (
            # If a checkpoint source IP is given
            checkpoint_node_ip
            # And the checkpoint does not currently exist on the local node
            and not os.path.exists(checkpoint_path)
            # And the source IP is different to the current IP
            and checkpoint_node_ip != ray.util.get_node_ip_address()
        ):
            checkpoint = _get_checkpoint_from_remote_node(
                checkpoint_path, checkpoint_node_ip
            )
            if checkpoint:
                checkpoint.to_directory(checkpoint_path)

        if not os.path.exists(checkpoint_path):
            if fallback_to_latest:
                logger.info(
                    f"Checkpoint path was not available, trying to recover from latest "
                    f"available checkpoint instead. Unavailable checkpoint path: "
                    f"{checkpoint_path}"
                )
                checkpoint_path = self._get_latest_available_checkpoint()
                if checkpoint_path:
                    logger.info(
                        f"Trying to recover from latest available checkpoint: "
                        f"{checkpoint_path}"
                    )
                    return self.restore(checkpoint_path, fallback_to_latest=False)

            # Else, raise
            raise ValueError(
                f"Could not recover from checkpoint as it does not exist on local "
                f"disk and was not available on cloud storage or another Ray node. "
                f"Got checkpoint path: {checkpoint_path} and IP {checkpoint_node_ip}"
            )

        checkpoint_dir = checkpoint_path
        metadata = TrainableUtil.load_metadata(checkpoint_dir)

        # Set metadata
        self._experiment_id = metadata["experiment_id"]
        self._iteration = metadata["iteration"]
        self._timesteps_total = metadata["timesteps_total"]
        self._time_total = metadata["time_total"]
        self._episodes_total = metadata["episodes_total"]

        # Actually load checkpoint
        self.load_checkpoint(checkpoint_dir)

        self._time_since_restore = 0.0
        self._timesteps_since_restore = 0
        self._iterations_since_restore = 0
        self._restored = True
        self._iteration = 0

        self._counters[NUM_AGENT_STEPS_SAMPLED] = 0
        self._counters[NUM_AGENT_STEPS_TRAINED] = 0
        self._counters[NUM_ENV_STEPS_SAMPLED] = 0
        self._counters[NUM_ENV_STEPS_TRAINED] = 0

        logger.info(
            "Restored on %s from checkpoint: %s", self._local_ip, checkpoint_dir
        )
        state = {
            "_iteration": self._iteration,
            "_timesteps_total": self._timesteps_total,
            "_time_total": self._time_total,
            "_episodes_total": self._episodes_total,
        }
        logger.info("Current state after restoring: %s", state)

    def load_checkpoint(self, checkpoint: Union[Dict, str]) -> None:
        # Checkpoint is provided as a directory name.
        # Restore from the checkpoint file or dir.
        if isinstance(checkpoint, str):
            checkpoint_info = get_checkpoint_info(checkpoint)
            checkpoint_data = Algorithm._checkpoint_info_to_algorithm_state(
                checkpoint_info
            )
        # Checkpoint is a checkpoint-as-dict -> Restore state from it as-is.
        else:
            checkpoint_data = checkpoint
        self.__setstate__(checkpoint_data)

    def save(
        self, checkpoint_dir: Optional[str] = None, prevent_upload: bool = False
    ) -> str:
        """Saves the current model state to a checkpoint.

        Subclasses should override ``save_checkpoint()`` instead to save state.
        This method dumps additional metadata alongside the saved path.

        If a remote checkpoint dir is given, this will also sync up to remote
        storage.

        Args:
            checkpoint_dir: Optional dir to place the checkpoint.
            prevent_upload: If True, will not upload the saved checkpoint to cloud.

        Returns:
            The given or created checkpoint directory.

        Note the return path should match up with what is expected of
        `restore()`.
        """
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        self.save_checkpoint(checkpoint_dir)
        
        metadata = self.get_state()

        metadata["relative_checkpoint_path"] = ""
        metadata["saved_as_dict"] = False

        TrainableUtil.write_metadata(checkpoint_dir, metadata)

        # Maybe sync to cloud
        if not prevent_upload:
            self._maybe_save_to_cloud(checkpoint_dir)

        return checkpoint_dir

    @PublicAPI
    def compute_actions(
        self,
        observations: TensorStructType,
        state: Optional[List[TensorStructType]] = None,
        *,
        prev_action: Optional[TensorStructType] = None,
        prev_reward: Optional[TensorStructType] = None,
        info: Optional[EnvInfoDict] = None,
        policy_id: PolicyID = DEFAULT_POLICY_ID,
        full_fetch: bool = False,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        episodes: Optional[List[Episode]] = None,
        unsquash_actions: Optional[bool] = None,
        clip_actions: Optional[bool] = None,
        # Deprecated.
        normalize_actions=None,
        **kwargs,
    ):
        """Computes an action for the specified policy on the local Worker.

        Note that you can also access the policy object through
        self.get_policy(policy_id) and call compute_actions() on it directly.

        Args:
            observation: Observation from the environment.
            state: RNN hidden state, if any. If state is not None,
                then all of compute_single_action(...) is returned
                (computed action, rnn state(s), logits dictionary).
                Otherwise compute_single_action(...)[0] is returned
                (computed action).
            prev_action: Previous action value, if any.
            prev_reward: Previous reward, if any.
            info: Env info dict, if any.
            policy_id: Policy to query (only applies to multi-agent).
            full_fetch: Whether to return extra action fetch results.
                This is always set to True if RNN state is specified.
            explore: Whether to pick an exploitation or exploration
                action (default: None -> use self.config["explore"]).
            timestep: The current (sampling) time step.
            episodes: This provides access to all of the internal episodes'
                state, which may be useful for model-based or multi-agent
                algorithms.
            unsquash_actions: Should actions be unsquashed according
                to the env's/Policy's action space? If None, use
                self.config["normalize_actions"].
            clip_actions: Should actions be clipped according to the
                env's/Policy's action space? If None, use
                self.config["clip_actions"].

        Keyword Args:
            kwargs: forward compatibility placeholder

        Returns:
            The computed action if full_fetch=False, or a tuple consisting of
            the full output of policy.compute_actions_from_input_dict() if
            full_fetch=True or we have an RNN-based Policy.
        """
        if normalize_actions is not None:
            deprecation_warning(
                old="Trainer.compute_actions(`normalize_actions`=...)",
                new="Trainer.compute_actions(`unsquash_actions`=...)",
                error=True,
            )
            unsquash_actions = normalize_actions

        # `unsquash_actions` is None: Use value of config['normalize_actions'].
        if unsquash_actions is None:
            unsquash_actions = self.config["normalize_actions"]
        # `clip_actions` is None: Use value of config['clip_actions'].
        elif clip_actions is None:
            clip_actions = self.config["clip_actions"]

        # Preprocess obs and states.
        state_defined = state is not None
        policy = self.get_policy(policy_id)
        filtered_obs, filtered_state = [], []
        for agent_id, ob in observations.items():
            worker = self.workers.local_worker()
            preprocessed = worker.preprocessors[policy_id].transform(ob)
            filtered = worker.filters[policy_id](preprocessed, update=False)
            filtered_obs.append(filtered)
            if state is None:
                continue
            elif agent_id in state:
                filtered_state.append(state[agent_id])
            else:
                filtered_state.append(policy.get_initial_state())

        # Batch obs and states
        obs_batch = np.stack(filtered_obs)
        if state is None:
            state = []
        else:
            state = list(zip(*filtered_state))
            state = [np.stack(s) for s in state]

        input_dict = {SampleBatch.OBS: obs_batch}

        # prev_action and prev_reward can be None, np.ndarray, or tensor-like structure.
        # Explicitly check for None here to avoid the error message "The truth value of
        # an array with more than one element is ambiguous.", when np arrays are passed
        # as arguments.
        if prev_action is not None:
            input_dict[SampleBatch.PREV_ACTIONS] = prev_action
        if prev_reward is not None:
            input_dict[SampleBatch.PREV_REWARDS] = prev_reward
        if info:
            input_dict[SampleBatch.INFOS] = info
        for i, s in enumerate(state):
            input_dict[f"state_in_{i}"] = s

        # Batch compute actions
        actions, states, infos = policy.compute_actions_from_input_dict(
            input_dict=input_dict,
            explore=explore,
            timestep=timestep,
            episodes=episodes,
        )

        # Unbatch actions for the environment into a multi-agent dict.
        single_actions = space_utils.unbatch(actions)
        actions = {}
        for key, a in zip(observations, single_actions):
            # If we work in normalized action space (normalize_actions=True),
            # we re-translate here into the env's action space.
            if unsquash_actions:
                a = space_utils.unsquash_action(a, policy.action_space_struct)
            # Clip, according to env's action space.
            elif clip_actions:
                a = space_utils.clip_action(a, policy.action_space_struct)
            actions[key] = a

        # Unbatch states into a multi-agent dict.
        unbatched_states = {}
        for idx, agent_id in enumerate(observations):
            unbatched_states[agent_id] = [s[idx] for s in states]

        # Return only actions or full tuple
        if state_defined or full_fetch:
            return actions, unbatched_states, infos
        else:
            return actions


    def compute_single_action_of_gan(
        self,
        observation: Optional[TensorStructType] = None,
        state: Optional[List[TensorStructType]] = None,
        *,
        prev_action: Optional[TensorStructType] = None,
        prev_reward: Optional[float] = None,
        info: Optional[EnvInfoDict] = None,
        input_dict: Optional[SampleBatch] = None,
        policy_id: PolicyID = DEFAULT_POLICY_ID,
        full_fetch: bool = False,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        episode: Optional[Episode] = None,
        unsquash_action: Optional[bool] = None,
        clip_action: Optional[bool] = None,
        # Deprecated args.
        unsquash_actions=DEPRECATED_VALUE,
        clip_actions=DEPRECATED_VALUE,
        # Kwargs placeholder for future compatibility.
        gen_task: Optional[int] = None,
        **kwargs,
    ) -> Union[
        TensorStructType,
        Tuple[TensorStructType, List[TensorType], Dict[str, TensorType]],
    ]:
        """Computes an action for the specified policy on the local worker.

        Note that you can also access the policy object through
        self.get_policy(policy_id) and call compute_single_action() on it
        directly.

        Args:
            observation: Single (unbatched) observation from the
                environment.
            state: List of all RNN hidden (single, unbatched) state tensors.
            prev_action: Single (unbatched) previous action value.
            prev_reward: Single (unbatched) previous reward value.
            info: Env info dict, if any.
            input_dict: An optional SampleBatch that holds all the values
                for: obs, state, prev_action, and prev_reward, plus maybe
                custom defined views of the current env trajectory. Note
                that only one of `obs` or `input_dict` must be non-None.
            policy_id: Policy to query (only applies to multi-agent).
                Default: "default_policy".
            full_fetch: Whether to return extra action fetch results.
                This is always set to True if `state` is specified.
            explore: Whether to apply exploration to the action.
                Default: None -> use self.config["explore"].
            timestep: The current (sampling) time step.
            episode: This provides access to all of the internal episodes'
                state, which may be useful for model-based or multi-agent
                algorithms.
            unsquash_action: Should actions be unsquashed according to the
                env's/Policy's action space? If None, use the value of
                self.config["normalize_actions"].
            clip_action: Should actions be clipped according to the
                env's/Policy's action space? If None, use the value of
                self.config["clip_actions"].

        Keyword Args:
            kwargs: forward compatibility placeholder

        Returns:
            The computed action if full_fetch=False, or a tuple of a) the
            full output of policy.compute_actions() if full_fetch=True
            or we have an RNN-based Policy.

        Raises:
            KeyError: If the `policy_id` cannot be found in this Trainer's
                local worker.
        """
        if clip_actions != DEPRECATED_VALUE:
            deprecation_warning(
                old="Trainer.compute_single_action(`clip_actions`=...)",
                new="Trainer.compute_single_action(`clip_action`=...)",
                error=True,
            )
            clip_action = clip_actions
        if unsquash_actions != DEPRECATED_VALUE:
            deprecation_warning(
                old="Trainer.compute_single_action(`unsquash_actions`=...)",
                new="Trainer.compute_single_action(`unsquash_action`=...)",
                error=True,
            )
            unsquash_action = unsquash_actions

        # `unsquash_action` is None: Use value of config['normalize_actions'].
        if unsquash_action is None:
            unsquash_action = self.config["normalize_actions"]
        # `clip_action` is None: Use value of config['clip_actions'].
        elif clip_action is None:
            clip_action = self.config["clip_actions"]

        # User provided an input-dict: Assert that `obs`, `prev_a|r`, `state`
        # are all None.
        err_msg = (
            "Provide either `input_dict` OR [`observation`, ...] as "
            "args to Trainer.compute_single_action!"
        )
        if input_dict is not None:
            assert (
                observation is None
                and prev_action is None
                and prev_reward is None
                and state is None
            ), err_msg
            observation = input_dict[SampleBatch.OBS]
        else:
            assert observation is not None, err_msg

        # Get the policy to compute the action for (in the multi-agent case,
        # Trainer may hold >1 policies).
        policy = self.get_policy(policy_id)
        if policy is None:
            raise KeyError(
                f"PolicyID '{policy_id}' not found in PolicyMap of the "
                f"Trainer's local worker!"
            )
        local_worker = self.workers.local_worker()

        # Check the preprocessor and preprocess, if necessary.
        pp = local_worker.preprocessors[policy_id]
        if pp and type(pp).__name__ != "NoPreprocessor":
            observation = pp.transform(observation)
        observation = local_worker.filters[policy_id](observation, update=False)

        # Input-dict.
        if input_dict is not None:
            input_dict[SampleBatch.OBS] = observation
            action = policy.compute_single_action_of_gan(
                input_dict=input_dict,
                explore=explore,
                timestep=timestep,
                episode=episode,
                gen_task=gen_task,
            )
        # Individual args.
        else:
            action = policy.compute_single_action_of_gan(
                obs=observation,
                state=state,
                prev_action=prev_action,
                prev_reward=prev_reward,
                info=info,
                explore=explore,
                timestep=timestep,
                episode=episode,
                gen_task=gen_task,
            )

        # If we work in normalized action space (normalize_actions=True),
        # we re-translate here into the env's action space.
        if unsquash_action:
            action = space_utils.unsquash_action(action, policy.action_space_struct)
        # Clip, according to env's action space.
        elif clip_action:
            action = space_utils.clip_action(action, policy.action_space_struct)

        return action




def train_one_step(algorithm, train_batch, policies_to_train=None, CL_option=None, lock_update=False) -> Dict:
    """Function that improves the all policies in `train_batch` on the local worker.

    Examples:
        >>> from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
        >>> algo = [...] # doctest: +SKIP
        >>> train_batch = synchronous_parallel_sample(algo.workers) # doctest: +SKIP
        >>> # This trains the policy on one batch.
        >>> results = train_one_step(algo, train_batch)) # doctest: +SKIP
        {"default_policy": ...}

    Updates the NUM_ENV_STEPS_TRAINED and NUM_AGENT_STEPS_TRAINED counters as well as
    the LEARN_ON_BATCH_TIMER timer of the `self` object.
    """

    config = algorithm.config
    workers = algorithm.workers
    local_worker = workers.local_worker()
    num_sgd_iter = config.get("num_sgd_iter", 1)
    sgd_minibatch_size = config.get("sgd_minibatch_size", 0)

    learn_timer = algorithm._timers[LEARN_ON_BATCH_TIMER]
    with learn_timer:
        # Subsample minibatches (size=`sgd_minibatch_size`) from the
        # train batch and loop through train batch `num_sgd_iter` times.
        if num_sgd_iter > 1 or sgd_minibatch_size > 0:
            info = do_minibatch_sgd(
                train_batch,
                {
                    pid: local_worker.get_policy(pid)
                    for pid in policies_to_train
                    or local_worker.get_policies_to_train(train_batch)
                },
                local_worker,
                num_sgd_iter,
                sgd_minibatch_size,
                [],
                CL_option,
                lock_update
            )
        # Single update step using train batch.
        else:
            info = learn_on_batch_worker(local_worker, train_batch, CL_option, lock_update)

    learn_timer.push_units_processed(train_batch.count)
    algorithm._counters[NUM_ENV_STEPS_TRAINED] += train_batch.count
    algorithm._counters[NUM_AGENT_STEPS_TRAINED] += train_batch.agent_steps()

    if algorithm.reward_estimators:
        info[DEFAULT_POLICY_ID]["off_policy_estimation"] = {}
        for name, estimator in algorithm.reward_estimators.items():
            info[DEFAULT_POLICY_ID]["off_policy_estimation"][name] = estimator.train(
                train_batch
            )
    return info

def multi_gpu_train_one_step(algorithm, train_batch, CL_option=None, lock_update=False) -> Dict:
    """Multi-GPU version of train_one_step.

    Uses the policies' `load_batch_into_buffer` and `learn_on_loaded_batch` methods
    to be more efficient wrt CPU/GPU data transfers. For example, when doing multiple
    passes through a train batch (e.g. for PPO) using `config.num_sgd_iter`, the
    actual train batch is only split once and loaded once into the GPU(s).

    Examples:
        >>> from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
        >>> algo = [...] # doctest: +SKIP
        >>> train_batch = synchronous_parallel_sample(algo.workers) # doctest: +SKIP
        >>> # This trains the policy on one batch.
        >>> results = multi_gpu_train_one_step(algo, train_batch)) # doctest: +SKIP
        {"default_policy": ...}

    Updates the NUM_ENV_STEPS_TRAINED and NUM_AGENT_STEPS_TRAINED counters as well as
    the LOAD_BATCH_TIMER and LEARN_ON_BATCH_TIMER timers of the Algorithm instance.
    """
    config = algorithm.config
    workers = algorithm.workers
    local_worker = workers.local_worker()
    num_sgd_iter = config.get("num_sgd_iter", 1)
    sgd_minibatch_size = config.get("sgd_minibatch_size", config["train_batch_size"])

    # Determine the number of devices (GPUs or 1 CPU) we use.
    num_devices = int(math.ceil(config["num_gpus"] or 1))

    # Make sure total batch size is dividable by the number of devices.
    # Batch size per tower.
    per_device_batch_size = sgd_minibatch_size // num_devices
    # Total batch size.
    batch_size = per_device_batch_size * num_devices
    assert batch_size % num_devices == 0
    assert batch_size >= num_devices, "Batch size too small!"

    # Handle everything as if multi-agent.
    train_batch = train_batch.as_multi_agent()

    # Load data into GPUs.
    load_timer = algorithm._timers[LOAD_BATCH_TIMER]
    with load_timer:
        num_loaded_samples = {}
        for policy_id, batch in train_batch.policy_batches.items():
            # Not a policy-to-train.
            if (
                local_worker.is_policy_to_train is not None
                and not local_worker.is_policy_to_train(policy_id, train_batch)
            ):
                continue

            # Decompress SampleBatch, in case some columns are compressed.
            batch.decompress_if_needed()

            # Load the entire train batch into the Policy's only buffer
            # (idx=0). Policies only have >1 buffers, if we are training
            # asynchronously.
            num_loaded_samples[policy_id] = local_worker.policy_map[
                policy_id
            ].load_batch_into_buffer(batch, buffer_index=0)

    # Execute minibatch SGD on loaded data.
    learn_timer = algorithm._timers[LEARN_ON_BATCH_TIMER]
    with learn_timer:
        # Use LearnerInfoBuilder as a unified way to build the final
        # results dict from `learn_on_loaded_batch` call(s).
        # This makes sure results dicts always have the same structure
        # no matter the setup (multi-GPU, multi-agent, minibatch SGD,
        # tf vs torch).
        learner_info_builder = LearnerInfoBuilder(num_devices=num_devices)

        for policy_id, samples_per_device in num_loaded_samples.items():
            policy = local_worker.policy_map[policy_id]
            num_batches = max(1, int(samples_per_device) // int(per_device_batch_size))
            logger.debug("== sgd epochs for {} ==".format(policy_id))
            for _ in range(num_sgd_iter):
                permutation = np.random.permutation(num_batches)
                for batch_index in range(num_batches):
                    # Learn on the pre-loaded data in the buffer.
                    # Note: For minibatch SGD, the data is an offset into
                    # the pre-loaded entire train batch.
                    results = learn_on_loaded_batch_policy(
                        policy, permutation[batch_index] * per_device_batch_size, buffer_index=0, CL_option=CL_option, lock_update=lock_update
                    )

                    learner_info_builder.add_learn_on_batch_results(results, policy_id)

        # Tower reduce and finalize results.
        learner_info = learner_info_builder.finalize()

    load_timer.push_units_processed(train_batch.count)
    learn_timer.push_units_processed(train_batch.count)

    # TODO: Move this into Trainer's `training_iteration` method for
    #  better transparency.
    algorithm._counters[NUM_ENV_STEPS_TRAINED] += train_batch.count
    algorithm._counters[NUM_AGENT_STEPS_TRAINED] += train_batch.agent_steps()

    if algorithm.reward_estimators:
        learner_info[DEFAULT_POLICY_ID]["off_policy_estimation"] = {}
        for name, estimator in algorithm.reward_estimators.items():
            learner_info[DEFAULT_POLICY_ID]["off_policy_estimation"][
                name
            ] = estimator.train(train_batch)

    return learner_info

def do_minibatch_sgd(
    samples,
    policies,
    local_worker,
    num_sgd_iter,
    sgd_minibatch_size,
    standardize_fields,
    CL_option=None,
    lock_update=False
):
    """Execute minibatch SGD.

    Args:
        samples: Batch of samples to optimize.
        policies: Dictionary of policies to optimize.
        local_worker: Master rollout worker instance.
        num_sgd_iter: Number of epochs of optimization to take.
        sgd_minibatch_size: Size of minibatches to use for optimization.
        standardize_fields: List of sample field names that should be
            normalized prior to optimization.

    Returns:
        averaged info fetches over the last SGD epoch taken.
    """

    # Handle everything as if multi-agent.
    samples = samples.as_multi_agent()

    # Use LearnerInfoBuilder as a unified way to build the final
    # results dict from `learn_on_loaded_batch` call(s).
    # This makes sure results dicts always have the same structure
    # no matter the setup (multi-GPU, multi-agent, minibatch SGD,
    # tf vs torch).
    learner_info_builder = LearnerInfoBuilder(num_devices=1)
    for policy_id, policy in policies.items():
        if policy_id not in samples.policy_batches:
            continue

        batch = samples.policy_batches[policy_id]
        for field in standardize_fields:
            batch[field] = standardized(batch[field])

        # Check to make sure that the sgd_minibatch_size is not smaller
        # than max_seq_len otherwise this will cause indexing errors while
        # performing sgd when using a RNN or Attention model
        if (
            policy.is_recurrent()
            and policy.config["model"]["max_seq_len"] > sgd_minibatch_size
        ):
            raise ValueError(
                "`sgd_minibatch_size` ({}) cannot be smaller than"
                "`max_seq_len` ({}).".format(
                    sgd_minibatch_size, policy.config["model"]["max_seq_len"]
                )
            )

        for i in range(num_sgd_iter):
            for minibatch in minibatches(batch, sgd_minibatch_size):
                results = (learn_on_batch_worker(local_worker, 
                                MultiAgentBatch({policy_id: minibatch}, minibatch.count), CL_option, lock_update
                           ))[policy_id]
                learner_info_builder.add_learn_on_batch_results(results, policy_id)

    learner_info = learner_info_builder.finalize()
    return learner_info

def learn_on_batch_worker(worker, samples: SampleBatchType, CL_option=None, lock_update=False) -> Dict:
    """Update policies based on the given batch.

    This is the equivalent to apply_gradients(compute_gradients(samples)),
    but can be optimized to avoid pulling gradients into CPU memory.

    Args:
        samples: The SampleBatch or MultiAgentBatch to learn on.

    Returns:
        Dictionary of extra metadata from compute_gradients().

    Examples:
        >>> import gym
        >>> from ray.rllib.evaluation.rollout_worker import RolloutWorker
        >>> from ray.rllib.algorithms.pg.pg_tf_policy import PGTF1Policy
        >>> worker = RolloutWorker( # doctest: +SKIP
        ...   env_creator=lambda _: gym.make("CartPole-v1"), # doctest: +SKIP
        ...   default_policy_class=PGTF1Policy) # doctest: +SKIP
        >>> batch = worker.sample() # doctest: +SKIP
        >>> info = worker.learn_on_batch(samples) # doctest: +SKIP
    """
    if log_once("learn_on_batch"):
        logger.info(
            "Training on concatenated sample batches:\n\n{}\n".format(
                summarize(samples)
            )
        )

    info_out = {}
    if isinstance(samples, MultiAgentBatch):
        builders = {}
        to_fetch = {}
        for pid, batch in samples.policy_batches.items():
            if worker.is_policy_to_train is not None and not worker.is_policy_to_train(
                pid, samples
            ):
                continue
            # Decompress SampleBatch, in case some columns are compressed.
            batch.decompress_if_needed()
            policy = worker.policy_map[pid]
            tf_session = policy.get_session()
            if tf_session and hasattr(policy, "_build_learn_on_batch"):
                builders[pid] = _TFRunBuilder(tf_session, "learn_on_batch")
                to_fetch[pid] = policy._build_learn_on_batch(builders[pid], batch, CL_option, lock_update)
            else:
                info_out[pid] = learn_on_batch_policy(policy, batch, CL_option, lock_update)
        info_out.update({pid: builders[pid].get(v) for pid, v in to_fetch.items()})
    else:
        if worker.is_policy_to_train is None or worker.is_policy_to_train(
            DEFAULT_POLICY_ID, samples
        ):
            info_out.update(
                {
                    DEFAULT_POLICY_ID: learn_on_batch_policy(worker.policy_map[DEFAULT_POLICY_ID], samples, CL_option, lock_update)
                }
            )
    if log_once("learn_out"):
        logger.debug("Training out:\n\n{}\n".format(summarize(info_out)))
    return info_out

def learn_on_batch_policy(policy, postprocessed_batch: SampleBatch, CL_option=None, lock_update=False) -> Dict[str, TensorType]:

    # Set Model to train mode.
    if policy.model:
        policy.model.train()
    # Callback handling.
    learn_stats = {}
    policy.callbacks.on_learn_on_batch(
        policy=policy, train_batch=postprocessed_batch, result=learn_stats
    )

    # Compute gradients (will calculate all losses and `backward()`
    # them to get the grads).
    grads, fetches = compute_gradients(policy, postprocessed_batch, CL_option)

    # Step the optimizers.
    if not lock_update:
        policy.apply_gradients(_directStepOptimizerSingleton)
        policy.num_grad_updates += 1

    if policy.model and hasattr(policy.model, "metrics"):
        fetches["model"] = policy.model.metrics()
    else:
        fetches["model"] = {}

    if not lock_update:
        fetches.update(
            {
                "custom_metrics": learn_stats,
                NUM_AGENT_STEPS_TRAINED: postprocessed_batch.count,
                NUM_GRAD_UPDATES_LIFETIME: policy.num_grad_updates,
                # -1, b/c we have to measure this diff before we do the update above.
                DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: (
                    policy.num_grad_updates
                    - 1
                    - (postprocessed_batch.num_grad_updates or 0)
                ),
            }
        )
    else:
        fetches.update(
            {
                "custom_metrics": learn_stats,
                NUM_AGENT_STEPS_TRAINED: 0,
                NUM_GRAD_UPDATES_LIFETIME: 0,
                # -1, b/c we have to measure this diff before we do the update above.
                DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: (
                    0
                ),
            }
        )

    return fetches

def learn_on_loaded_batch_policy(policy, offset: int = 0, buffer_index: int = 0, CL_option=None, lock_update=False):
    if not policy._loaded_batches[buffer_index]:
        raise ValueError(
            "Must call Policy.load_batch_into_buffer() before "
            "Policy.learn_on_loaded_batch()!"
        )

    # Get the correct slice of the already loaded batch to use,
    # based on offset and batch size.
    device_batch_size = policy.config.get(
        "sgd_minibatch_size", policy.config["train_batch_size"]
    ) // len(policy.devices)

    # Set Model to train mode.
    if policy.model_gpu_towers:
        for t in policy.model_gpu_towers:
            t.train()

    # Shortcut for 1 CPU only: Batch should already be stored in
    # `self._loaded_batches`.
    if len(policy.devices) == 1 and policy.devices[0].type == "cpu":
        assert buffer_index == 0
        if device_batch_size >= len(policy._loaded_batches[0][0]):
            batch = policy._loaded_batches[0][0]
        else:
            batch = policy._loaded_batches[0][0][offset : offset + device_batch_size]
        return policy.learn_on_batch(batch)

    if len(policy.devices) > 1:
        # Copy weights of main model (tower-0) to all other towers.
        state_dict = policy.model.state_dict()
        # Just making sure tower-0 is really the same as self.model.
        assert policy.model_gpu_towers[0] is policy.model
        for tower in policy.model_gpu_towers[1:]:
            tower.load_state_dict(state_dict)

    if device_batch_size >= sum(len(s) for s in policy._loaded_batches[buffer_index]):
        device_batches = policy._loaded_batches[buffer_index]
    else:
        device_batches = [
            b[offset : offset + device_batch_size]
            for b in policy._loaded_batches[buffer_index]
        ]

    # Callback handling.
    batch_fetches = {}
    for i, batch in enumerate(device_batches):
        custom_metrics = {}
        policy.callbacks.on_learn_on_batch(
            policy=policy, train_batch=batch, result=custom_metrics
        )
        batch_fetches[f"tower_{i}"] = {"custom_metrics": custom_metrics}

    # Do the (maybe parallelized) gradient calculation step.
    if lock_update:
        CL_option_ = None
    else:
        CL_option_ = CL_option
    tower_outputs = _multi_gpu_parallel_grad_calc(policy, device_batches, CL_option_)

    # Mean-reduce gradients over GPU-towers (do this on CPU: self.device).
    all_grads = []
    for i in range(len(tower_outputs[0][0])):
        if tower_outputs[0][0][i] is not None:
            all_grads.append(
                torch.mean(
                    torch.stack([t[0][i].to(policy.device) for t in tower_outputs]),
                    dim=0,
                )
            )
        else:
            all_grads.append(None)
    # Set main model's grads to mean-reduced values.
    for i, p in enumerate(policy.model.parameters()):
        p.grad = all_grads[i]

    if not lock_update:
        policy.apply_gradients(_directStepOptimizerSingleton)
        policy.num_grad_updates += 1

    for i, (model, batch) in enumerate(zip(policy.model_gpu_towers, device_batches)):
        if not lock_update:
            batch_fetches[f"tower_{i}"].update(
                {
                    LEARNER_STATS_KEY: policy.stats_fn(batch),
                    "model": model.metrics(),
                    NUM_GRAD_UPDATES_LIFETIME: policy.num_grad_updates,
                    # -1, b/c we have to measure this diff before we do the update
                    # above.
                    DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: (
                        policy.num_grad_updates - 1 - (batch.num_grad_updates or 0)
                    ),
                }
            )
        else:
            batch_fetches[f"tower_{i}"].update(
                {
                    LEARNER_STATS_KEY: policy.stats_fn(batch),
                    "model": model.metrics(),
                    NUM_GRAD_UPDATES_LIFETIME: 0,
                    # -1, b/c we have to measure this diff before we do the update
                    # above.
                    DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: (
                        0
                    ),
                }
            )
    
    batch_fetches.update(policy.extra_compute_grad_fetches())

    return batch_fetches

def compute_gradients(policy, postprocessed_batch: SampleBatch, CL_option=None) -> ModelGradients:

    assert len(policy.devices) == 1

    # If not done yet, see whether we have to zero-pad this batch.
    if not postprocessed_batch.zero_padded:
        pad_batch_to_sequences_of_same_size(
            batch=postprocessed_batch,
            max_seq_len=policy.max_seq_len,
            shuffle=False,
            batch_divisibility_req=policy.batch_divisibility_req,
            view_requirements=policy.view_requirements,
        )

    postprocessed_batch.set_training(True)
    policy._lazy_tensor_dict(postprocessed_batch, device=policy.devices[0])

    # Do the (maybe parallelized) gradient calculation step.
    tower_outputs = _multi_gpu_parallel_grad_calc(policy, [postprocessed_batch], CL_option)

    all_grads, grad_info = tower_outputs[0]

    grad_info["allreduce_latency"] /= len(policy._optimizers)
    grad_info.update(policy.stats_fn(postprocessed_batch))

    fetches = policy.extra_compute_grad_fetches()

    return all_grads, dict(fetches, **{LEARNER_STATS_KEY: grad_info})

def _multi_gpu_parallel_grad_calc(
    policy, sample_batches: List[SampleBatch], CL_option=None
) -> List[Tuple[List[TensorType], GradInfoDict]]:
    """Performs a parallelized loss and gradient calculation over the batch.

    Splits up the given train batch into n shards (n=number of this
    Policy's devices) and passes each data shard (in parallel) through
    the loss function using the individual devices' models
    (policy.model_gpu_towers). Then returns each tower's outputs.

    Args:
        sample_batches: A list of SampleBatch shards to
            calculate loss and gradients for.

    Returns:
        A list (one item per device) of 2-tuples, each with 1) gradient
        list and 2) grad info dict.
    """
    assert len(policy.model_gpu_towers) == len(sample_batches)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(shard_idx, model, sample_batch, device):
        torch.set_grad_enabled(grad_enabled)
        try:
            with NullContextManager() if device.type == "cpu" else torch.cuda.device(  # noqa: E501
                device
            ):
                loss_out = force_list(
                    policy.loss(model, policy.dist_class, sample_batch, CL_option)
                )

                # Call Model's custom-loss with Policy loss outputs and
                # train_batch.
                if hasattr(model, "custom_loss"):
                    loss_out = model.custom_loss(loss_out, sample_batch)

                assert len(loss_out) == len(policy._optimizers)

                # Loop through all optimizers.
                grad_info = {"allreduce_latency": 0.0}

                parameters = list(model.parameters())
                all_grads = [None for _ in range(len(parameters))]
                for opt_idx, opt in enumerate(policy._optimizers):
                    # Erase gradients in all vars of the tower that this
                    # optimizer would affect.
                    param_indices = policy.multi_gpu_param_groups[opt_idx]
                    for param_idx, param in enumerate(parameters):
                        if param_idx in param_indices and param.grad is not None:
                            param.grad.data.zero_()
                    # Recompute gradients of loss over all variables.
                    loss_out[opt_idx].backward(retain_graph=True)
                    grad_info.update(
                        policy.extra_grad_process(opt, loss_out[opt_idx])
                    )

                    grads = []
                    # Note that return values are just references;
                    # Calling zero_grad would modify the values.
                    for param_idx, param in enumerate(parameters):
                        if param_idx in param_indices:
                            if param.grad is not None:
                                grads.append(param.grad)
                            all_grads[param_idx] = param.grad

                    if policy.distributed_world_size:
                        start = time.time()
                        if torch.cuda.is_available():
                            # Sadly, allreduce_coalesced does not work with
                            # CUDA yet.
                            for g in grads:
                                torch.distributed.all_reduce(
                                    g, op=torch.distributed.ReduceOp.SUM
                                )
                        else:
                            torch.distributed.all_reduce_coalesced(
                                grads, op=torch.distributed.ReduceOp.SUM
                            )

                        for param_group in opt.param_groups:
                            for p in param_group["params"]:
                                if p.grad is not None:
                                    p.grad /= policy.distributed_world_size

                        grad_info["allreduce_latency"] += time.time() - start

            with lock:
                results[shard_idx] = (all_grads, grad_info)
        except Exception as e:
            import traceback

            with lock:
                results[shard_idx] = (
                    ValueError(
                        e.args[0]
                        + "\n traceback"
                        + traceback.format_exc()
                        + "\n"
                        + "In tower {} on device {}".format(shard_idx, device)
                    ),
                    e,
                )

    # Single device (GPU) or fake-GPU case (serialize for better
    # debugging).
    if len(policy.devices) == 1 or policy.config["_fake_gpus"]:
        for shard_idx, (model, sample_batch, device) in enumerate(
            zip(policy.model_gpu_towers, sample_batches, policy.devices)
        ):
            _worker(shard_idx, model, sample_batch, device)
            # Raise errors right away for better debugging.
            last_result = results[len(results) - 1]
            if isinstance(last_result[0], ValueError):
                raise last_result[0] from last_result[1]
    # Multi device (GPU) case: Parallelize via threads.
    else:
        threads = [
            threading.Thread(
                target=_worker, args=(shard_idx, model, sample_batch, device)
            )
            for shard_idx, (model, sample_batch, device) in enumerate(
                zip(policy.model_gpu_towers, sample_batches, policy.devices)
            )
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    # Gather all threads' outputs and return.
    outputs = []
    for shard_idx in range(len(sample_batches)):
        output = results[shard_idx]
        if isinstance(output[0], Exception):
            raise output[0] from output[1]
        outputs.append(results[shard_idx])
    return outputs


