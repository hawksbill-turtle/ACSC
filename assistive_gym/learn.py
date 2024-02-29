## The code is written based on Assistive-Gym (https://github.com/Healthcare-Robotics/assistive-gym)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from assistive_gym.agents import ppo_gan

import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob, math
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
from distutils.dir_util import copy_tree

from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
import torch, copy


TRAIN_BATCH_SIZE = 19200

def memory_minibatches(samples: SampleBatch, mem_minibatch_size: int):
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


def setup_config(env, algo, coop=False, seed=0, framework='torch', extra_configs={}, custom_configs={}, fisher=False, EWC_task_count=0, num_task=0, evaluation_envs=[], ewc_coeff=1.0, gen_coeff=1.0, dis_coeff=1.0, etc_coeff1=1.0, etc_coeff2=1.0, etc_coeff3=1.0, save_memory=False, load_memory=False, memory_root='./memory', save_memory_len=1000000, load_memory_len=100000, memory_portion=0.2):
    num_processes = multiprocessing.cpu_count()
    if algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        if not fisher:
            config['train_batch_size'] = TRAIN_BATCH_SIZE
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

    config['gen_coeff'] = gen_coeff
    config['dis_coeff'] = dis_coeff

    config['etc_coeff1'] = etc_coeff1
    config['etc_coeff2'] = etc_coeff2
    config['etc_coeff3'] = etc_coeff3

    config['save_memory'] = save_memory
    config['load_memory'] = load_memory
    config['memory_root'] = memory_root
    config['save_memory_len'] = save_memory_len
    config['load_memory_len'] = load_memory_len
    config['memory_portion'] = memory_portion

    if config['load_memory']:
        assert config['memory_root'] is not None, "memory file should not None"
        assert config['memory_portion'] is not None and config['memory_portion'] >= 0 and config['memory_portion'] <= 1, "memory portion should between 0 and 1"
        config['memory_files'] = []
        for env in config['eval_envs']:
            memory_fn = 'memory_' + env[14:-3] + '.pt'
            memory_fn = os.path.join(config['memory_root'], memory_fn)
            config['memory_files'].append(memory_fn)
            assert os.path.exists(memory_fn), "memory file should exists"
        if not fisher:
            config['memory_batch_size'] = math.ceil(config['train_batch_size'] * config['memory_portion'])
            config['train_batch_size'] -= config['memory_batch_size']

    if num_task != 0:
        current_loop = EWC_task_count//num_task
        config['lr'] = config['lr'] * (0.8**current_loop)
    
    return {**config, **extra_configs}

def load_policy(env, algo, env_name, policy_path=None, coop=False, seed=0, framework='torch', extra_configs={}, custom_configs={}, fisher=False, EWC_task_count=0, num_task=0, evaluation_envs=[], ewc_coeff=1.0, gen_coeff=1.0, dis_coeff=1.0, etc_coeff1=1.0, etc_coeff2=1.0, etc_coeff3=1.0, save_memory=False, load_memory=False, memory_root='./memory', save_memory_len=1000000, load_memory_len=100000, memory_portion=0.2):
    if algo == 'ppo':
        agent = ppo_gan.PPO_GANTrainer(setup_config(env, algo, coop, seed, framework, extra_configs, custom_configs, fisher=fisher, EWC_task_count=EWC_task_count, num_task=num_task, evaluation_envs=evaluation_envs, ewc_coeff=ewc_coeff, gen_coeff=gen_coeff, dis_coeff=dis_coeff, etc_coeff1=etc_coeff1, etc_coeff2=etc_coeff2, etc_coeff3=etc_coeff3, save_memory=save_memory, load_memory=load_memory, memory_root=memory_root, save_memory_len=save_memory_len, load_memory_len=load_memory_len, memory_portion=memory_portion), 'assistive_gym:'+env_name)
        agent.add_state_dict_keys(EWC_task_count)
    if policy_path != '':
        if os.path.exists(policy_path):
            agent.restore(policy_path)
            agent.set_EWC_num(EWC_task_count, num_task)
            return agent
    agent.set_EWC_num(EWC_task_count, num_task)
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

def train(env_name, algo, timesteps_total=1000000, save_dir='./trained_models/', load_policy_path='', coop=False, seed=0, extra_configs={}, framework='torch', custom_configs={}, EWC_task_count=0, num_task=0, eval_reward={}, evaluation_envs=[], ewc_coeff=1.0, gen_coeff=1.0, dis_coeff=1.0, etc_coeff1=1.0, etc_coeff2=1.0, etc_coeff3=1.0, save_memory=False, load_memory=False, memory_root='./memory', save_memory_len=1000000, load_memory_len=100000, memory_portion=0.2, memory_generator=None):
    env = make_env(env_name, coop, seed)
    agent = load_policy(env, algo, env_name, load_policy_path, coop, seed, framework, extra_configs, custom_configs, EWC_task_count=EWC_task_count, num_task=num_task, evaluation_envs=evaluation_envs, ewc_coeff=ewc_coeff, gen_coeff=gen_coeff, dis_coeff=dis_coeff, etc_coeff1=etc_coeff1, etc_coeff2=etc_coeff2, etc_coeff3=etc_coeff3, save_memory=save_memory, load_memory=load_memory, memory_root=memory_root, save_memory_len=save_memory_len, load_memory_len=load_memory_len, memory_portion=memory_portion)
    
    timesteps = 0
    while timesteps < timesteps_total:
        result = agent.train(CL_option='EWC', memory_generator=memory_generator)
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
    agent.cleanup()

def update_fisher(env_name, algo, timesteps_total=1000000, save_dir='./trained_models/', load_policy_path='', coop=False, seed=0, extra_configs={}, framework='torch', custom_configs={}, EWC_task_count=0, num_task=0, evaluation_envs=[], ewc_epochs=100, ewc_steps=20, gen_coeff=1.0, dis_coeff=1.0, etc_coeff1=1.0, etc_coeff2=1.0, etc_coeff3=1.0):
    env = make_env(env_name, coop, seed)
    agent = load_policy(env, algo, env_name, load_policy_path, coop, seed, framework, extra_configs, custom_configs, fisher=True, EWC_task_count=EWC_task_count, num_task=num_task, evaluation_envs=evaluation_envs, gen_coeff=gen_coeff, dis_coeff=dis_coeff, etc_coeff1=etc_coeff1, etc_coeff2=etc_coeff2, etc_coeff3=etc_coeff3)
    agent.set_EWC_num(EWC_task_count, num_task)

    result = agent.update_fisher(ewc_epochs=ewc_epochs, num_ewc_steps=ewc_steps)

    # Save the recently trained policy
    if save_dir != '':
        agent.save(save_dir)
        agent.save(os.path.join(save_dir, env_name+'_'+str(EWC_task_count//num_task)+'_EWC'))
    env.disconnect()
    agent.cleanup()

def render_policy(env, env_name, algo, policy_path, coop=False, colab=False, seed=0, n_episodes=1, extra_configs={}):
    if env is None:
        env = make_env(env_name, coop, seed=seed)
        if colab:
            env.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs)

    if not colab:
        env.render()
    frames = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            if coop:
                # Compute the next action for the robot/human using the trained policies
                action_robot = test_agent.compute_action(obs['robot'], policy_id='robot')
                action_human = test_agent.compute_action(obs['human'], policy_id='human')
                # Step the simulation forward using the actions from our trained policies
                obs, reward, done, info = env.step({'robot': action_robot, 'human': action_human})
                done = done['__all__']
            else:
                # Compute the next action using the trained policy
                action = test_agent.compute_action(obs)
                # Step the simulation forward using the action from our trained policy
                obs, reward, done, info = env.step(action)
            if colab:
                # Capture (render) an image from the camera
                img, depth = env.get_camera_image_depth()
                frames.append(img)
    env.disconnect()
    agent.cleanup()
    if colab:
        filename = 'output_%s.png' % env_name
        write_apng(filename, frames, delay=100)
        return filename

def evaluate_policy(env_name, algo, policy_path, n_episodes=100, coop=False, seed=0, verbose=False, extra_configs={}):
    env = make_env(env_name, coop, seed=seed)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs)

    rewards = []
    forces = []
    task_successes = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        reward_total = 0.0
        force_list = []
        task_success = 0.0
        while not done:
            if coop:
                # Compute the next action for the robot/human using the trained policies
                action_robot = test_agent.compute_action(obs['robot'], policy_id='robot')
                action_human = test_agent.compute_action(obs['human'], policy_id='human')
                # Step the simulation forward using the actions from our trained policies
                obs, reward, done, info = env.step({'robot': action_robot, 'human': action_human})
                reward = reward['robot']
                done = done['__all__']
                info = info['robot']
            else:
                action = test_agent.compute_action(obs)
                obs, reward, done, info = env.step(action)
            reward_total += reward
            force_list.append(info['total_force_on_human'])
            task_success = info['task_success']

        rewards.append(reward_total)
        forces.append(np.mean(force_list))
        task_successes.append(task_success)
        if verbose:
            print('Reward total: %.2f, mean force: %.2f, task success: %r' % (reward_total, np.mean(force_list), task_success))
        sys.stdout.flush()
    test_agent.cleanup()
    env.disconnect()

    print('\n', '-'*50, '\n')
    # print('Rewards:', rewards)
    print('Reward Mean:', np.mean(rewards))
    print('Reward Std:', np.std(rewards))

    # print('Forces:', forces)
    print('Force Mean:', np.mean(forces))
    print('Force Std:', np.std(forces))

    # print('Task Successes:', task_successes)
    print('Task Success Mean:', np.mean(task_successes))
    print('Task Success Std:', np.std(task_successes))
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
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
    parser.add_argument('--num-loop', type=int, default=4,
                        help='Number of simulation timesteps to train a policy (default: 1000000)')
    parser.add_argument('--ewc-coeff', type=float, default=1.0,
                        help='EWC loss coefficient')
    parser.add_argument('--ewc-epochs', type=int, default=100,
                        help='EWC epochs')
    parser.add_argument('--ewc-steps', type=int, default=40,
                        help='EWC steps')
    parser.add_argument('--gen-coeff', type=float, default=1.0,
                        help='GAN loss coefficient for generator')
    parser.add_argument('--dis-coeff', type=float, default=1.0,
                        help='GAN loss coefficient for discriminator')
    parser.add_argument('--etc-coeff1', type=float, default=1.0,
                        help='ACSC loss coefficient')
    parser.add_argument('--etc-coeff2', type=float, default=1.0,
                        help='ACSC loss coefficient')
    parser.add_argument('--etc-coeff3', type=float, default=1.0,
                        help='ACSC loss coefficient')
    parser.add_argument('--save-memory', action='store_true', default=False,
                        help='Whether to save memory')
    parser.add_argument('--load-memory', action='store_true', default=True,
                        help='Whether to load memory')
    parser.add_argument('--memory-root', default='./memory',
                        help='Where to save memory')
    parser.add_argument('--save-memory-len', type=int, default=1000000,
                        help='Length of memory')
    parser.add_argument('--load-memory-len', type=int, default=100000,
                        help='Length of memory')
    parser.add_argument('--memory-portion', type=float, default=0.2,
                        help='Number of simulation timesteps to train a policy (default: 1000000)')
    args = parser.parse_args()

    ModelCatalog.register_custom_model("PPO_GAN_ModelV2", PPO_GAN_ModelV2)

    ## possible tasks: ScratchItch, BedBathing, Feeding, Drinking, Dressing, and ArmManipulation
    ## possible robots: PR2, Jaco, Baxter, Sawyer, Stretch, Panda
    ## tasks = ['ScratchItch', 'BedBathing', 'Feeding', 'Drinking', 'Dressing', 'ArmManipulation']
    tasks_ = ['Feeding', 'ScratchItch', 'BedBathing', 'Drinking']
    robot = 'Panda'

    custom_configs = {
        "custom_model": "PPO_GAN_ModelV2",
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {'num_task':len(tasks_)},
    }
    ewc = True
    
    tasks = [t+robot+'-v1' for t in tasks_]
    num_loop = args.num_loop
    train_timesteps = args.train_timesteps

    EWC_task_count = 0
    num_task = len(tasks)
    
    seed = args.seed
    
    ewc_coeff = args.ewc_coeff
    ewc_epochs = args.ewc_epochs
    ewc_steps = args.ewc_steps

    gen_coeff = args.gen_coeff
    dis_coeff = args.dis_coeff

    etc_coeff1 = args.etc_coeff1
    etc_coeff2 = args.etc_coeff2
    etc_coeff3 = args.etc_coeff3

    save_memory = args.save_memory
    load_memory = args.load_memory
    memory_root = args.memory_root
    save_memory_len = args.save_memory_len
    load_memory_len = args.load_memory_len
    memory_portion = args.memory_portion

    if load_memory:
        memory_from_start = False
        memory_files = []
        for env in tasks:
            memory_fn = 'memory_' + env[:-3] + '.pt'
            memory_fn = os.path.join(memory_root, memory_fn)
            memory_files.append(memory_fn)
            assert os.path.exists(memory_fn), "memory file should exists"

        memories = []            
        for idx, mfn in enumerate(memory_files):
            memory_ = torch.load(mfn)
            if memory_from_start:
                memory = copy.deepcopy(memory_[:min(len(memory_), load_memory_len)])
            else:
                memory = copy.deepcopy(memory_[max(0, len(memory_)-load_memory_len):])
            del memory_
            memory['task_idx'] = np.array([idx for _ in range(len(memory))])
            memories.append(memory)
        
        memories = SampleBatch.concat_samples(memories)
        memories['from_memory'] = np.array([1 for _ in range(len(memories))])
        memory_batch_size = math.ceil(TRAIN_BATCH_SIZE * memory_portion)

    save_dir = os.path.join(args.save_dir,str(datetime.now())+'_'+args.algo+'_'+str(tasks_)+'_'+robot+'_nl_'+str(num_loop)+'_tt_'+str(train_timesteps)+'_ec_'+str(ewc_coeff)+'_ee_'+str(ewc_epochs)+'_es_'+str(ewc_steps)+'_gc_'+str(gen_coeff)+'_dc_'+str(dis_coeff)+'_ec1_'+str(etc_coeff1)+'_ec2_'+str(etc_coeff2)+'_ec3_'+str(etc_coeff3)+'_lml_'+str(load_memory_len)+'_mp_'+str(memory_portion))
    load_policy_path = save_dir

    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    
    eval_reward = {}
    eval_reward['timesteps'] = [] 
    for _type in (['mean', 'max', 'min', 'rewards']):
        eval_reward[_type] = {}
        for idx in range(num_task):
            eval_reward[_type]['env' + str(idx)] = []

    for loop in range(num_loop):
        for task in tasks:
            args.env = task
            coop = ('Human' in args.env)
            if args.train:
                memory_generator = memory_minibatches(memories, memory_batch_size)
                train(args.env, args.algo, timesteps_total=args.train_timesteps, save_dir=save_dir, load_policy_path=load_policy_path, coop=coop, seed=seed, framework=args.framework, custom_configs=custom_configs, EWC_task_count=EWC_task_count, num_task=num_task, eval_reward=eval_reward, evaluation_envs=tasks, ewc_coeff=ewc_coeff, gen_coeff=gen_coeff, dis_coeff=dis_coeff, etc_coeff1=etc_coeff1, etc_coeff2=etc_coeff2, etc_coeff3=etc_coeff3, save_memory=save_memory, load_memory=load_memory, memory_root=memory_root, save_memory_len=save_memory_len, load_memory_len=load_memory_len, memory_portion=memory_portion, memory_generator=memory_generator)
            if args.render:
                render_policy(None, args.env, args.algo, checkpoint_path if checkpoint_path is not None else load_policy_path, coop=coop, colab=args.colab, seed=seed, n_episodes=args.render_episodes)
            if args.evaluate:
                evaluate_policy(args.env, args.algo, checkpoint_path if checkpoint_path is not None else load_policy_path, n_episodes=args.eval_episodes, coop=coop, seed=seed, verbose=args.verbose)
            
            seed += 1

            if ewc:
                update_fisher(args.env, args.algo, timesteps_total=args.train_timesteps, save_dir=save_dir, load_policy_path=load_policy_path, coop=coop, seed=seed, framework=args.framework, custom_configs=custom_configs, EWC_task_count=EWC_task_count, num_task=num_task, evaluation_envs=tasks, ewc_epochs=ewc_epochs, ewc_steps=ewc_steps, gen_coeff=gen_coeff, dis_coeff=dis_coeff, etc_coeff1=etc_coeff1, etc_coeff2=etc_coeff2, etc_coeff3=etc_coeff3)
            
            EWC_task_count += 1
            seed += 1

