# -*- coding: utf-8 -*-
from __future__ import division
import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch

from collections import deque

import numpy as np
import matplotlib.pylab as plt
import scipy.io
import math

DEFAULT_PLOTLY_COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

REWARD_RANGE=(-300, 200)


def find_results(data_path, target_len=126):
    dirs = os.listdir(data_path)
    data_dirs = []; data_envs = []; data_robots = []; 
    data_gc = []; data_dc = []; ## gc: generator coeff, dc: discriminator epochs
    data_ec1 = []; data_ec2 = []; data_ec3 = []; ## ec: etc coeff, 1: same task, 2: value, 3: action
    data_lml = []; data_mp = []

    for d in dirs:
        if '[' in d and 'lml_' in d and 'mp_' in d:
            envs = d[d.index('[')+2:d.index(']')-1].split("\', \'")
            robot = d[d.index(']')+2:d.index('nl_')-1]
            gc = float(d[d.index('gc_')+len('gc_'):d.index('dc_')-1])
            dc = float(d[d.index('dc_')+len('dc_'):d.index('ec1_')-1])
            ec1 = float(d[d.index('ec1_')+len('ec1_'):d.index('ec2_')-1])
            ec2 = float(d[d.index('ec2_')+len('ec2_'):d.index('ec3_')-1])
            ec3 = float(d[d.index('ec3_')+len('ec3_'):d.index('lml_')-1])
            lml = float(d[d.index('lml_')+len('lml_'):d.index('mp_')-1])
            mp = float(d[d.index('mp_')+len('mp_'):])
            
            mn = os.path.join(data_path, d, 'aa_result.mat')
            if os.path.exists(mn):
                results = scipy.io.loadmat(mn)['eval_reward'][0][0][0][0][1][0][0][0][0]
            
                if len(results) == target_len:
                    data_dirs.append(d); data_envs.append(envs); data_robots.append(robot)
                    data_gc.append(gc); data_dc.append(dc)
                    data_ec1.append(ec1); data_ec2.append(ec2); data_ec3.append(ec3)
                    data_lml.append(lml); data_mp.append(mp)
    
    return data_dirs, data_envs, data_robots, data_gc, data_dc, data_ec1, data_ec2, data_ec3, data_lml, data_mp

def data_for_plot(r, tag, log_step, color_idx):
    ys_mean = torch.tensor(r[0])
    xs = torch.tensor([(j+1)*19200*log_step for j, y in enumerate(ys_mean)])

    trace_mean = Scatter(x=xs, y=ys_mean.cpu().numpy(), line=Line(color=DEFAULT_PLOTLY_COLORS[color_idx], width=1), name=tag)
    
    return trace_mean

def shape_for_plot(r, log_step, env_idx, num_env):
    a = 42
   
    ys_mean = torch.tensor(r[0])
    xs = torch.tensor([(j+1)*19200*log_step for j, y in enumerate(ys_mean)])

    shape = []
    j = 0
    while j*19200 < xs[-1]:
        if ((j/a)%num_env) == env_idx:
            shape.append(dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=(j)*19200,
                        y0=REWARD_RANGE[0],
                        x1=(j+a)*19200,
                        y1=REWARD_RANGE[1],
                        fillcolor='rgb(255, 40, 40)',
                        opacity=0.1,
                        line_width=0,
                        layer="below"
                      ))
        j += a
    
    return shape

def plot_reward(data, shape, title, path=''):  
    plotly.offline.plot({
      'data': data,
      'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title}, shapes=shape)
    }, filename=os.path.join(path, title+'.html'), auto_open=False)



if __name__=="__main__":
    data_path = '../trained_models/31-1/trained_models/target'
    result_path = os.path.join(data_path, 'results')

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    
    train_timesteps = 800000
    step = 19200
    eval_step = 3
    loop = 3
    num_task = 3
    
    target_len = math.ceil(math.ceil(train_timesteps/step) / eval_step) * loop * num_task
    
    data_dirs, data_envs, data_robots, data_gc, data_dc, data_ec1, data_ec2, data_ec3, data_lml, data_mp = find_results(data_path, target_len=target_len)
    
    data = [[] for i in range(len(data_envs[0]))]; shapes = []
    
    log_step = 3
    if log_step == None:
        log_step = int(input('Log step? '))
    
    num_env = 3
    
    for j, dd in enumerate(data_dirs):
        mn = os.path.join(data_path, dd, 'aa_result.mat')
        results = scipy.io.loadmat(mn)['eval_reward'][0][0][0][0][1][0][0]
        for i, r in enumerate(results):
            tag = 'ec1 ' + str(data_ec1[j]) + ' ec2 ' + str(data_ec2[j]) + ' ec3 ' + str(data_ec3[j])
            data[i].append(data_for_plot(r, tag, log_step, j))
            if len(shapes) < i+1:
                shapes.append(shape_for_plot(r, log_step, i, num_env))
    
    for i in range(len(data_envs[0])):
        title = data_envs[0][i] + ' ewc epoch ' + str(data_gc[0]) + ' ewc step ' + str(data_dc[0])
        plot_reward(data[i], shapes[i], title, result_path)



















