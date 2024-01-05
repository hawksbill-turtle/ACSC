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
from ray.rllib.execution.train_ops import (
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

from ray.tune.trainable import Trainable
from ray.rllib.utils.annotations import DeveloperAPI
from ray.air._internal.util import skip_exceptions, exception_cause

from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_SAMPLED_THIS_ITER,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_THIS_ITER,
    NUM_ENV_STEPS_TRAINED,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TRAINING_ITERATION_TIMER,
)

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

from ray.air.checkpoint import (
    Checkpoint,
    _DICT_CHECKPOINT_ADDITIONAL_FILE_KEY,
)

import time
from datetime import datetime

from ray.rllib.evaluation.worker_set import WorkerSet

from ray.rllib.evaluation.metrics import (
    collect_episodes,
    collect_metrics,
    summarize_episodes,
)

from ray.rllib.execution.common import (
    AGENT_STEPS_TRAINED_COUNTER,
    APPLY_GRADS_TIMER,
    COMPUTE_GRADS_TIMER,
    LAST_TARGET_UPDATE_TS,
    LEARN_ON_BATCH_TIMER,
    LOAD_BATCH_TIMER,
    NUM_TARGET_UPDATES,
    STEPS_SAMPLED_COUNTER,
    STEPS_TRAINED_COUNTER,
    STEPS_TRAINED_THIS_ITER_COUNTER,
    _check_sample_batch_type,
    _get_global_vars,
    _get_shared_metrics,
)

from ray.rllib.utils.typing import (
    AgentID,
    EnvCreator,
    EnvType,
    ModelGradients,
    ModelWeights,
    MultiAgentPolicyConfigDict,
    PartialAlgorithmConfigDict,
    PolicyID,
    PolicyState,
    SampleBatchType,
    T,
    AlgorithmConfigDict,
    GradInfoDict,
    PolicyState,
    TensorStructType,
    TensorType,
)

from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.rllib.utils.sgd import minibatches

from ray.rllib.utils.debug import summarize, update_global_seed_if_necessary
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size

from ray.rllib.utils import NullContextManager, force_list

import threading
import torch

from ray.rllib.policy.torch_policy import _directStepOptimizerSingleton

from ray.tune.trainable.util import TrainableUtil

from ray.rllib.utils.metrics import (
    DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY,
    NUM_AGENT_STEPS_TRAINED,
    NUM_GRAD_UPDATES_LIFETIME,
)

from ray.rllib.utils.checkpoints import CHECKPOINT_VERSION, get_checkpoint_info

import os

from collections import defaultdict

import psutil

logger = logging.getLogger(__name__)

def train_(agent, CL_option=None, memory_generator=None):
    """Runs one logical iteration of training.

    Calls ``step()`` internally. Subclasses should override ``step()``
    instead to return results.
    This method automatically fills the following fields in the result:

        `done` (bool): training is terminated. Filled only if not provided.

        `time_this_iter_s` (float): Time in seconds this iteration
        took to run. This may be overridden in order to override the
        system-computed time difference.

        `time_total_s` (float): Accumulated time in seconds for this
        entire experiment.

        `experiment_id` (str): Unique string identifier
        for this experiment. This id is preserved
        across checkpoint / restore calls.

        `training_iteration` (int): The index of this
        training iteration, e.g. call to train(). This is incremented
        after `step()` is called.

        `pid` (str): The pid of the training process.

        `date` (str): A formatted date of when the result was processed.

        `timestamp` (str): A UNIX timestamp of when the result
        was processed.

        `hostname` (str): Hostname of the machine hosting the training
        process.

        `node_ip` (str): Node ip of the machine hosting the training
        process.

    Returns:
        A dict that describes training progress.
    """
    if agent._warmup_time is None:
        agent._warmup_time = time.time() - agent._start_time
    start = time.time()
    try:
        result = agent.step(CL_option, lock_update=False, memory_generator=memory_generator)
    except Exception as e:
        skipped = skip_exceptions(e)
        raise skipped from exception_cause(skipped)

    assert isinstance(result, dict), "step() needs to return a dict."
    
    # We do not modify internal state nor update this result if duplicate.
    if RESULT_DUPLICATE in result:
        return result

    result = result.copy()

    agent._iteration += 1
    agent._iterations_since_restore += 1

    if result.get(TIME_THIS_ITER_S) is not None:
        time_this_iter = result[TIME_THIS_ITER_S]
    else:
        time_this_iter = time.time() - start
    agent._time_total += time_this_iter
    agent._time_since_restore += time_this_iter

    result.setdefault(DONE, False)

    # agent._timesteps_total should only be tracked if increments provided
    if result.get(TIMESTEPS_THIS_ITER) is not None:
        if agent._timesteps_total is None:
            agent._timesteps_total = 0
        agent._timesteps_total += result[TIMESTEPS_THIS_ITER]
        agent._timesteps_since_restore += result[TIMESTEPS_THIS_ITER]

    # agent._episodes_total should only be tracked if increments provided
    if result.get(EPISODES_THIS_ITER) is not None:
        if agent._episodes_total is None:
            agent._episodes_total = 0
        agent._episodes_total += result[EPISODES_THIS_ITER]

    # agent._timesteps_total should not override user-provided total
    result.setdefault(TIMESTEPS_TOTAL, agent._timesteps_total)
    result.setdefault(EPISODES_TOTAL, agent._episodes_total)
    result.setdefault(TRAINING_ITERATION, agent._iteration)

    # Provides auto-filled neg_mean_loss for avoiding regressions
    if result.get("mean_loss"):
        result.setdefault("neg_mean_loss", -result["mean_loss"])

    now = datetime.today()
    result.update(agent.get_auto_filled_metrics(now, time_this_iter))

    monitor_data = agent._monitor.get_data()
    if monitor_data:
        result.update(monitor_data)

    agent.log_result(result)

    if agent._stdout_context:
        agent._stdout_stream.flush()
    if agent._stderr_context:
        agent._stderr_stream.flush()
    
    agent._last_result = result

    return result


def update_fisher_(agent, fisher_dict):
    """Runs one logical iteration of training.

    Calls ``step()`` internally. Subclasses should override ``step()``
    instead to return results.
    This method automatically fills the following fields in the result:

        `done` (bool): training is terminated. Filled only if not provided.

        `time_this_iter_s` (float): Time in seconds this iteration
        took to run. This may be overridden in order to override the
        system-computed time difference.

        `time_total_s` (float): Accumulated time in seconds for this
        entire experiment.

        `experiment_id` (str): Unique string identifier
        for this experiment. This id is preserved
        across checkpoint / restore calls.

        `training_iteration` (int): The index of this
        training iteration, e.g. call to train(). This is incremented
        after `step()` is called.

        `pid` (str): The pid of the training process.

        `date` (str): A formatted date of when the result was processed.

        `timestamp` (str): A UNIX timestamp of when the result
        was processed.

        `hostname` (str): Hostname of the machine hosting the training
        process.

        `node_ip` (str): Node ip of the machine hosting the training
        process.

    Returns:
        A dict that describes training progress.
    """
    est_fisher_info = fisher_dict
    
    if agent._warmup_time is None:
        agent._warmup_time = time.time() - agent._start_time
    start = time.time()
    try:
        result = agent.step(CL_option=None, lock_update=True)
    except Exception as e:
        skipped = skip_exceptions(e)
        raise skipped from exception_cause(skipped)
    
    assert isinstance(result, dict), "step() needs to return a dict."

    for n, p in agent.get_policy().model.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')
            if p.grad is not None:
                est_fisher_info[n] += p.grad.detach() ** 2

    for i, opt in enumerate(agent.get_policy()._optimizers):
        opt.zero_grad()

    # We do not modify internal state nor update this result if duplicate.
    if RESULT_DUPLICATE in result:
        return result

    result = result.copy()

    if result.get(TIME_THIS_ITER_S) is not None:
        time_this_iter = result[TIME_THIS_ITER_S]
    else:
        time_this_iter = time.time() - start
    agent._time_total += time_this_iter
    agent._time_since_restore += time_this_iter

    result.setdefault(DONE, False)

    # agent._timesteps_total should only be tracked if increments provided
    if result.get(TIMESTEPS_THIS_ITER) is not None:
        agent._timesteps_total = 0

    # agent._episodes_total should only be tracked if increments provided
    if result.get(EPISODES_THIS_ITER) is not None:
        agent._episodes_total = 0

    # agent._timesteps_total should not override user-provided total
    result.setdefault(TIMESTEPS_TOTAL, agent._timesteps_total)
    result.setdefault(EPISODES_TOTAL, agent._episodes_total)
    result.setdefault(TRAINING_ITERATION, agent._iteration)

    # Provides auto-filled neg_mean_loss for avoiding regressions
    if result.get("mean_loss"):
        result.setdefault("neg_mean_loss", -result["mean_loss"])

    now = datetime.today()
    result.update(agent.get_auto_filled_metrics(now, time_this_iter))

    monitor_data = agent._monitor.get_data()
    if monitor_data:
        result.update(monitor_data)

    agent.log_result(result)

    if agent._stdout_context:
        agent._stdout_stream.flush()
    if agent._stderr_context:
        agent._stderr_stream.flush()

    agent._last_result = result

    return result, est_fisher_info
