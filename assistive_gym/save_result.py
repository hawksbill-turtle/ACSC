import os
import numpy as np
import scipy.io as sio

def save_result(save_dir, log_name, result, eval_reward, num_task):
    eval_reward['timesteps'].append(result['timesteps_total'])
    for idx in range(num_task):
        eval_reward['mean']['env'+str(idx)].append(result['info']['learner']['evaluation'+str(idx)]['episode_reward_mean'])
        eval_reward['max']['env'+str(idx)].append(result['info']['learner']['evaluation'+str(idx)]['episode_reward_max'])
        eval_reward['min']['env'+str(idx)].append(result['info']['learner']['evaluation'+str(idx)]['episode_reward_min'])
        eval_reward['rewards']['env'+str(idx)].append(result['info']['learner']['evaluation'+str(idx)]['hist_stats']['episode_reward'])
    
    sio.savemat(os.path.join(save_dir, log_name + '_result.mat'), {'eval_reward':np.array(eval_reward)})
