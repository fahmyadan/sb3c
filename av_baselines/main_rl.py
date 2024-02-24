import os
import sys
from pathlib import Path 
import argparse 
import yaml 
import gymnasium as gym
from sb3.stable_baselines3.common.env_util import make_vec_env
from sb3.stable_baselines3.ppo.ppo import PPO
from sb3.stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from sb3c.sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
import numpy as np
import torch
from collections import OrderedDict
import wandb
from wandb.integration.sb3 import WandbCallback
from copy import deepcopy


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers and a string.')

    # Add arguments
    parser.add_argument('--number', type=int, help='An integer number', default=0)
    parser.add_argument('--text', type=str, help='A simple text string', default='fahmy')
    parser.add_argument('--env_config', type=str, help='path to gym environment config', default='env_cfg.yml')


    # Parse the arguments
    args = parser.parse_args()

    return args


def parse_yaml_file(file_path):
    """
    Parses the contents of a YAML file. All yml files should be in config

    :param file_path: The path to the YAML file to be parsed.
    :return: The data parsed from the YAML file.
    """

    path_to_cfgs = os.path.join(str(Path(__file__).parents[0]), 'configs/')
    file_path = path_to_cfgs + file_path

    with open(file_path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)
            return None

def update_env_config(vec_env, env_kwargs):

    for env in vec_env.envs:

        env.unwrapped.configure(env_kwargs)
    
    return vec_env

def update_env_spaces(vec_env):

    for env in vec_env.envs:

        env.define_spaces()
    
    vec_env.observation_space = env.observation_space

def update_vec_buffers(vec_env):

    vec_env.keys, shapes, dtypes = obs_space_info(vec_env.observation_space)
    vec_env.buf_obs = OrderedDict([(k, np.zeros((vec_env.num_envs, *tuple(shapes[k])), dtype=dtypes[k])) for k in vec_env.keys])

def wanb_init(alg_cfg, log_cfg):
    env_name = {'env_name':env_id}
    alg_cfg = deepcopy(alg_cfg)
    alg_cfg.update(env_name)

    run = wandb.init(project=log_cfg['project_name'], config=alg_cfg, sync_tensorboard=True, monitor_gym=True)

    return run 

if __name__ == "__main__":
    args = parse_arguments()
    env_cfg = parse_yaml_file(args.env_config)
    global env_id
    env_id = env_cfg['env_id']
    env_kwargs = env_cfg['env_args']
    logging_args = env_cfg['logging']

    if logging_args['wandb']:
        run = wanb_init(env_cfg['alg_cfg'], logging_args)
        log_dir = logging_args['tensorboard_log']
        log_dir = log_dir + f'/{run.id}'  
    else: 
        run = None 
        log_dir = None

    vec_env = make_vec_env(env_id, n_envs=env_cfg['n_envs'])
    vec_env = update_env_config(vec_env, env_kwargs)
    update_env_spaces(vec_env)
    update_vec_buffers(vec_env)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RecurrentPPO(env=vec_env, verbose=2, policy_kwargs= env_cfg['policy_cfg'], device=device,tensorboard_log=log_dir, **env_cfg['alg_cfg'])
  
    if run: 
        model_save_path = f"models/{run.id}"
        wandb_cb = WandbCallback(
        gradient_save_freq=100,
        model_save_path=model_save_path,
        verbose=2,
    )
    else:
        wandb_cb = None 
        model_save_path = None 


    model.learn(total_timesteps=env_cfg['total_timesteps'],callback=wandb_cb)
    if run is not None: 
        run.finish()

    print('check')