## The code is written based on Assistive-Gym (https://github.com/Healthcare-Robotics/assistive-gym)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from assistive_gym.agents import ppo_gan

import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
from ray.rllib.agents import ppo
# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from numpngw import write_apng

from datetime import datetime

from assistive_gym.save_result import save_result

from ray.rllib.models import ModelCatalog
from assistive_gym.models.ppo_gan.PPO_GAN_ModelV2 import PPO_GAN_ModelV2

from ray.rllib.offline import DatasetWriter

from shutil import copyfile


def setup_config(env, algo, coop=False, seed=0, framework='torch', extra_configs={}, custom_configs={}, fisher=False, EWC_task_count=0, num_task=0, evaluation_envs=[], ewc_coeff=5000, save_memory=False, memory_root='./memory', save_memory_len=2000000):
    num_processes = multiprocessing.cpu_count()
    if algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        if not fisher:
            config['train_batch_size'] = 19200
            config['sgd_minibatch_size'] = 128
        else:
            config['train_batch_size'] = num_processes
            config['sgd_minibatch_size'] = num_processes
        config['num_sgd_iter'] = 50
        config['lambda'] = 0.95
        config['model']['fcnet_hiddens'] = [100, 100]
    config['num_workers'] = num_processes
    config['num_cpus_per_worker'] = 0
    config['seed'] = seed
    config['log_level'] = 'ERROR'
    if coop:
        obs = env.reset()
        policies = {'robot': (None, env.observation_space_robot, env.action_space_robot, {}), 'human': (None, env.observation_space_human, env.action_space_human, {})}
        config['multiagent'] = {'policies': policies, 'policy_mapping_fn': lambda a: a}
        config['env_config'] = {'num_agents': 2}
    
    if custom_configs != {}:
        config['model']['custom_model'] = custom_configs['custom_model']
        if 'custom_model_config' in custom_configs:
            config['model']['custom_model_config'] = custom_configs['custom_model_config']   

    config['framework'] = framework

    config['evaluation_interval'] = 3
    
    config['EWC_task_count'] = EWC_task_count
    config['num_task'] = num_task
    
    config['eval_envs'] = ['assistive_gym:'+ee for ee in evaluation_envs]
    
    config['ewc_coeff'] = ewc_coeff
    config['gen_coeff'] = 0
    config['dis_coeff'] = 0

    config['etc_coeff1'] = 0
    config['etc_coeff2'] = 0
    config['etc_coeff3'] = 0
    
    config['save_memory'] = save_memory
    config['load_memory'] = False
    config['memory_root'] = memory_root
    config['save_memory_len'] = save_memory_len

    if num_task != 0:
        current_loop = EWC_task_count//num_task
        config['lr'] = config['lr'] * (0.8**current_loop)
    
    return {**config, **extra_configs}

def load_policy(env, algo, env_name, policy_path=None, coop=False, seed=0, framework='torch', extra_configs={}, custom_configs={}, num_task=0, evaluation_envs=[], save_memory=False, memory_root='./memory', save_memory_len=2000000):
    if algo == 'ppo':
        agent = ppo_gan.PPO_GANTrainer(setup_config(env, algo, coop, seed, framework, extra_configs, custom_configs, fisher=False, EWC_task_count=0, num_task=num_task, evaluation_envs=evaluation_envs, ewc_coeff=0, save_memory=save_memory, memory_root=memory_root, save_memory_len=save_memory_len), 'assistive_gym:'+env_name)
        agent.add_state_dict_keys(0)
    if policy_path != '':
        if os.path.exists(policy_path):
            agent.restore(policy_path)
            agent.set_EWC_num(0, num_task)
            return agent
    agent.set_EWC_num(0, num_task)
    return agent

def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    return env

def train(env_name, algo, timesteps_total=1000000, save_dir='./trained_models/', load_policy_path='', coop=False, seed=0, extra_configs={}, framework='torch', custom_configs={}, EWC_task_count=0, num_task=0, eval_reward={}, evaluation_envs=[], ewc_coeff=5000, save_memory=False, memory_root='./memory', save_memory_len=1000000):
    env = make_env(env_name, coop, seed)
    agent = load_policy(env, algo, env_name, load_policy_path, coop, seed, framework, extra_configs, custom_configs, num_task=num_task, evaluation_envs=evaluation_envs, save_memory=save_memory, memory_root=memory_root, save_memory_len=save_memory_len)
    
    timesteps = 0
    while timesteps < timesteps_total:
        result = agent.train(CL_option='EWC')
        timesteps = result['timesteps_total']
        if coop:
            # Rewards are added in multi agent envs, so we divide by 2 since agents share the same reward in coop
            result['episode_reward_mean'] /= 2
            result['episode_reward_min'] /= 2
            result['episode_reward_max'] /= 2
        print(f"Loop count: {EWC_task_count//num_task}, Task index: {agent.evaluation_envs.index('assistive_gym:'+env_name)}, Iteration: {result['training_iteration']}, total timesteps: {result['timesteps_total']}, total time: {result['time_total_s']:.1f}, FPS: {result['timesteps_total']/result['time_total_s']:.1f}, mean reward: {result['episode_reward_mean']:.1f}, min/max reward: {result['episode_reward_min']:.1f}/{result['episode_reward_max']:.1f}")
        if 'evaluation0' in result['info']['learner']:
            save_result(save_dir, 'aa', result, eval_reward, num_task)
        sys.stdout.flush()

        # Save the recently trained policy
        if save_dir != '':
            agent.save(save_dir)
            agent.save(os.path.join(save_dir, env_name+'_'+str(EWC_task_count//num_task)))
    env.disconnect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--seed', type=int, default=7777,
                        help='Random seed (default: 7777)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train a new policy')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate a trained policy over n_episodes')
    parser.add_argument('--train-timesteps', type=int, default=1000000,
                        help='Number of simulation timesteps to train a policy (default: 1000000)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    parser.add_argument('--render-episodes', type=int, default=1,
                        help='Number of rendering episodes (default: 1)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--colab', action='store_true', default=False,
                        help='Whether rendering should generate an animated png rather than open a window (e.g. when using Google Colab)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to output more verbose prints')
    parser.add_argument('--framework', default='torch',
                        help='Whether to use pytorch of tensorflow')
    parser.add_argument('--save-memory', action='store_true', default=True,
                        help='Whether to save memory')
    parser.add_argument('--memory-root', default='./memory',
                        help='Where to save memory')
    parser.add_argument('--save-memory-len', type=int, default=1000000,
                        help='Length of memory')
    args = parser.parse_args()

    ModelCatalog.register_custom_model("PPO_GAN_ModelV2", PPO_GAN_ModelV2)

    custom_configs = {
        "custom_model": "PPO_GAN_ModelV2",
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {'num_task':1},
    }

    ## possible tasks: ScratchItch, BedBathing, Feeding, Drinking, Dressing, and ArmManipulation
    ## possible robots: PR2, Jaco, Baxter, Sawyer, Stretch, Panda
    ## tasks = ['ScratchItch', 'BedBathing', 'Feeding', 'Drinking', 'Dressing', 'ArmManipulation']
    tasks_ = ['ScratchItch', 'BedBathing', 'Feeding', 'Drinking']
    robot = 'Panda'
    
    tasks = [t+robot+'-v1' for t in tasks_]
    
    seed = args.seed

    save_memory = args.save_memory
    memory_root = args.memory_root
    save_memory_len = args.save_memory_len
    
    if not os.path.exists(memory_root):
        os.mkdir(memory_root)

    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)

    for task in tasks:
        args.env = task
        save_dir = os.path.join(args.save_dir,str(datetime.now())+'_'+args.algo+'_'+str(task)+'_'+robot)
        coop = ('Human' in args.env)

        eval_reward = {}
        eval_reward['timesteps'] = [] 
        for _type in (['mean', 'max', 'min', 'rewards']):
            eval_reward[_type] = {}
            for idx in range(1):
                eval_reward[_type]['env' + str(idx)] = []
        
        if args.train:
            train(args.env, args.algo, timesteps_total=args.train_timesteps, save_dir=save_dir, coop=coop, seed=seed, framework=args.framework, custom_configs=custom_configs, num_task=1, eval_reward=eval_reward, evaluation_envs=[task], save_memory=save_memory, memory_root=memory_root, save_memory_len=save_memory_len)
        
        seed += 1

