import gym, sys, argparse
import numpy as np
from .learn import make_env, ppo_gan, multiprocessing, ppo, ModelCatalog
import os, sys
from shutil import copyfile

## python3 -m assistive_gym.env_viewer

if sys.version_info < (3, 0):
    print('Please use Python 3')
    exit()

def setup_config(env, algo, coop=False, seed=0, framework='torch', extra_configs={}, custom_configs={}, fisher=False, EWC_task_count=0, num_task=0, evaluation_envs=[], ewc_coeff=1.0, gen_coeff=1.0, dis_coeff=1.0, etc_coeff1=1.0, etc_coeff2=1.0, etc_coeff3=1.0):
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

    config['gen_coeff'] = gen_coeff
    config['dis_coeff'] = dis_coeff

    config['etc_coeff1'] = etc_coeff1
    config['etc_coeff2'] = etc_coeff2
    config['etc_coeff3'] = etc_coeff3

    if num_task != 0:
        current_loop = EWC_task_count//num_task
        config['lr'] = config['lr'] * (0.8**current_loop)
    
    return {**config, **extra_configs}

def sample_action(env, coop):
    if coop:
        return {'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
    return env.action_space.sample()

def viewer(env, agent):
    while True:
        done = False
        env.render()
        observation = env.reset()
        while not done:
            action = agent.compute_single_action(observation)
            observation, reward, done, info = env.step(action)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='BedBathingPanda-v1',
                        help='Environment to test (default: ScratchItchPanda-v1)')
    args = parser.parse_args()

    policy_path = './trained_models/2023-04-20 02:58:43.956087_ppo_[\'Feeding\', \'ScratchItch\', \'BedBathing\']_Panda_num_loop_3_train_timesteps_800000_ewc_coeff_1.0_ewc_epochs_50_ewc_steps_40_gen_coeff_10.0_dis_coeff_10.0_etc_coeff1_1.0_etc_coeff2_0.1_etc_coeff3_1.0'

    if 'PPO_GAN_ModelV2.py' in os.listdir(policy_path):
        copyfile(os.path.join(policy_path, 'PPO_GAN_ModelV2.py'), './cache/PPO_GAN_ModelV2.py')
        from cache.PPO_GAN_ModelV2 import PPO_GAN_ModelV2
    elif 'PPO_GAN_ModelV2.py' in os.listdir(os.path.abspath(os.path.join(policy_path, os.pardir))):
        parent_dir = os.path.abspath(os.path.join(policy_path, os.pardir))
        copyfile(os.path.join(parent_dir, 'PPO_GAN_ModelV2.py'), './cache/PPO_GAN_ModelV2.py')
        from cache.PPO_GAN_ModelV import PPO_GAN_ModelV2
    else:
        from assistive_gym.models.ppo_gan.PPO_GAN_ModelV2 import PPO_GAN_ModelV2

    env_name = args.env
    coop = False
    seed = 3
    algo = 'ppo'
    framework = 'torch'
    extra_configs={}
    ModelCatalog.register_custom_model("PPO_GAN_ModelV2", PPO_GAN_ModelV2)
    custom_configs = {
        "custom_model": "PPO_GAN_ModelV2",
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {'num_task':3},
    }
    fisher=False

    tasks_ = ['Feeding', 'ScratchItch', 'BedBathing']
    robot = 'Panda'
    tasks = [t+robot+'-v1' for t in tasks_]

    EWC_task_count = 0
    num_task = len(tasks)

    env = make_env(env_name)

    agent = ppo_gan.PPO_GANTrainer(setup_config(env, algo, coop, seed, framework, extra_configs, custom_configs, fisher=fisher, EWC_task_count=EWC_task_count, num_task=num_task, evaluation_envs=tasks), 'assistive_gym:'+env_name)
    agent.restore(policy_path)
    viewer(env, agent)



