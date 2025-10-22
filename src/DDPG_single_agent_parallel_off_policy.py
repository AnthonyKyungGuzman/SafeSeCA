import sys
import gym
import numpy as np
import random
import torch
import os

import matplotlib.pyplot as plt
from utils import *

import argparse
import os
import time
from datetime import datetime

import pybullet as p

import gym_panda_frite
from database_frite import Database_Frite

from gym_panda_frite.envs.environment import Environment

from gym_panda_frite.envs.panda_frite_env_wrapper_reward_max import PandaFriteEnvWrapperRewardMax

from DDPG.ddpg import DDPGagent

####################### Added ################################
from SeCAAgent import SeCAAgent
from SeCA.controllers import REGISTRY as mac_REGISTRY
from pathlib import Path
import yaml
from SeCA.utils.logging import get_logger, Logger
_logger = get_logger()
import torch as th
from types import SimpleNamespace as SN
from os.path import dirname, abspath
import collections
import wandb
from wandb import AlertLevel
import math
from utils import MovingAverageReward
import torch.multiprocessing as mp
from queue import Empty
from functools import partial
from VecEnv import VecEnv
##############################################################
from json_decoder import JsonDecoder

parser = argparse.ArgumentParser()
# parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test' or 'test_real' or 'generate_db'
parser.add_argument('--gui', default=False, type=bool) # use gui to see graphic results
parser.add_argument('--config_file', default='./configs/default/default.json', type=str)
parser.add_argument('--algo_config_file', default='./SeCA/config/algs/seca', type=str)

args = parser.parse_args()


wandb.login()
wandb.tensorboard.patch()
wandb_group = "ddpg_single_agent_off_policy"

def args_sanity_check(config):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        print("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")
    return config
	


def get_params():
    if not os.path.isfile(args.config_file):
        raise RuntimeError("=> Config file JSON to load does not exit : " + args.config_file)
        return

    sim_args = SN()
    sim_args.args = args
    
    sim_args.json_decoder = JsonDecoder(args.config_file)
    json_decoder = sim_args.json_decoder

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
	
    sim_args.env_name = json_decoder.config_data["env"]["name"]
    sim_args.env_random_seed = json_decoder.config_data["env"]["random_seed"]
    sim_args.env_time_set_action = json_decoder.config_data["env"]["time_set_action"]
    sim_args.env_reward_type = json_decoder.config_data["env"]["reward_parameters"]["reward_index"]
	
    print("** ENV PARAMETERS **")
    print("env_name = {}".format(sim_args.env_name))
    print("env_random_seed = {}".format(sim_args.env_random_seed))
    print("env_time_set_action = {}".format(sim_args.env_time_set_action))
	
    sim_args.cuda = json_decoder.config_data["ddpg"]["cuda"]
    sim_args.max_memory_size = json_decoder.config_data["ddpg"]["max_memory_size"]
    sim_args.batch_size = json_decoder.config_data["ddpg"]["batch_size"]
    sim_args.log_interval = json_decoder.config_data["ddpg"]["log_interval"]
	
    print("** DDPG PARAMETERS **")
    print("ddpg_cuda = {}".format(sim_args.cuda))
    print("ddpg_batch_size = {}".format(sim_args.batch_size))
    print("ddpg_max_memory_size = {}".format(sim_args.max_memory_size))
    print("ddpg_log_interval = {}".format(sim_args.log_interval))
	
    sim_args.log_name = json_decoder.config_data["log"]["name"]
    sim_args.database_name = json_decoder.config_data["database"]["name"]

    sim_args.n_episodes = json_decoder.config_data["env_train"]["n_episodes"]
    sim_args.n_steps = json_decoder.config_data["env_train"]["n_steps"]
    
    
    print("** ENV MODE TRAIN OPTIONS **")
    print("n_episodes = {}".format(sim_args.n_episodes))
    print("n_steps = {}".format(sim_args.n_steps))
    # print("ddpg_load = {}".format(sim_args.load))
	
    print("** LOG PARAMETERS **")
    print("log_name = {}".format(sim_args.log_name))
	
    sim_args.rank = 0 
    sim_args.size = 1 

	########################## SeCA Params
	# Get the defaults from ddpg_config.yaml
    with open(os.path.join(os.path.dirname(__file__), "DDPG/", "ddpg_config.yaml"), "r") as f:
        try:
            ddpg_config_dict = yaml.load(f,Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    _config = args_sanity_check(ddpg_config_dict)
	# Save to disk by default for sacred

    sim_args.ddpg_args = SN(**_config)
    sim_args.ddpg_args.device = "cuda" if sim_args.ddpg_args.use_cuda else "cpu"    

	# configure tensorboard logger
    unique_token = "{}__{}__{}".format(sim_args.ddpg_args.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), sim_args.rank)
    sim_args.ddpg_args.unique_token = unique_token
    tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "src/DDPG/results", "tb_logs")
    tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)

    tb_exp_direc_path = Path(tb_exp_direc)
    tb_exp_direc_path.mkdir(exist_ok=True)

    sim_args.tb_exp_direc = tb_exp_direc
    sim_args.wandb_run = wandb.init(project="ddpg_dlo",
                        name=f"worker_{0}",
                        group=wandb_group,
                        config={"worker": 0, "total_processes": sim_args.ddpg_args.n_workers},
                        sync_tensorboard=False,
                        dir=tb_exp_direc  # <- this makes it easier for wandb to find your logs
                    )
    log_dir = os.path.join("runs", wandb.run.id)
    print(f"[DEBUG] Logging to {log_dir}")
	
    sim_args.wandb_run_dir = wandb.run.dir  # example: 'wandb/run-20250411_XXXXXX-xxxxxx'

    # Compose the path to output.log
    sim_args.wandb_output_log_path = os.path.join(sim_args.wandb_run_dir, 'output.log')

	#######################################
				
    if not os.path.isfile(json_decoder.config_dir_name  + sim_args.database_name):
        raise RuntimeError("=> Database file to load does not exit : " + json_decoder.config_dir_name  + sim_args.database_name)
        return

    sim_args.configsParentDir = Path(json_decoder.config_dir_name)
    sim_args.logsFileDir = str(sim_args.configsParentDir.parent) + "/logs/"

    sim_args.load = json_decoder.config_data["ddpg"]["load"]
    sim_args.load_from_dir = json_decoder.config_data["ddpg"]["load_from_dir"]
    sim_args.weights_folder = json_decoder.config_data["ddpg"]["weights_folder"] 

    sim_args.weights_dir = ""

	

    return sim_args

def DDPG_setup(params):

    params.json_decoder.config_data["all_spaces"]["action_space"]["index"] = params.ddpg_args.action_index

    params.json_decoder.config_data["ddpg"]["load"] = params.ddpg_args.load
    params.load = params.ddpg_args.load

    torch.manual_seed(params.env_random_seed + 0)
    
    if (params.ddpg_args.use_cuda):
        torch.cuda.manual_seed(params.env_random_seed + 0)
	
	#Params for DDPG Agent
    params.state_shape = params.ddpg_args.n_states
    params.n_actions = params.ddpg_args.n_actions

    params.ddpg_args.max_memory_size = params.max_memory_size
    
    params.ddpg_args.batch_size = params.batch_size
    params.ddpg_args.state_shape = params.ddpg_args.n_states 
    params.ddpg_args.episode_steps = params.json_decoder.config_data["env_train"]["n_steps"]

    # #select latest weights to load to nets
    if params.rank==0:
        if params.load:
            directory = str(params.configsParentDir.parent) + "/SeCA/weights/" 
            if params.load_from_dir:
                params.weights_dir = directory + params.weights_folder
            else:
                folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
                weights_dir = max(folders, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
                params.weights_dir = str(params.configsParentDir.parent) + "/SeCA/weights/" + weights_dir
        else: #create new folder for weights
            weights_dir = str(params.configsParentDir.parent) + "/SeCA/weights/" + "{}__{}".format(params.ddpg_args.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) 
            params.weights_dir = Path(weights_dir)
            params.weights_dir.mkdir(exist_ok=True)

    params.ddpg_args.weights_dir = str(params.weights_dir)+'/'

    return params


def create_test_env(sim_params):
    env_pybullet = Environment(json_decoder=sim_params.json_decoder, gui=sim_params.args.gui)
    env_pybullet.startThread()
    db = Database_Frite(json_decoder=sim_params.json_decoder)
    env = PandaFriteEnvWrapperRewardMax(gym.make(sim_params.env_name, database=db, json_decoder = sim_params.json_decoder, env_pybullet=env_pybullet, gui=sim_params.args.gui, env_rank=0))
    env.seed(sim_params.env_random_seed + 0)

    sim_params.n_steps_test = sim_params.json_decoder.config_data["env_test"]["n_steps"]
    sim_params.test = SN()
    sim_params.test.test_return_ma = MovingAverageReward(window_size=100)
    sim_params.test.test_num = 0

    return env


def test_env(env, main_agent, sim_params):

    msg = '=> [{}] Testing agent:'.format(datetime.now())
    print(msg)

    env.reset_env_bullet()
    panda_num = 0
    state, initial_error = env.reset_bullet(panda_num)

    episode_reward = 0

    for step in range(sim_params.n_steps_test):
        action = main_agent.get_action(state, test_mode=True)
        next_state, rewards, done, info = env.step_bullet(action,panda_num,0,0,step)

        state = next_state
        episode_reward += rewards

        if done:
            break

    sim_params.test.test_return_ma.add_reward(episode_reward)
    sim_params.test.test_num += 1 

    test_msg = '=> [{}] Test finished:  test_num: {}, return: {:.4f}, steps: {}, return_ma: {:.4f}'.format(datetime.now(),  sim_params.test.test_num, episode_reward, step, sim_params.test.test_return_ma.get_average())
    print(test_msg)

    sim_params.wandb_run.log({"test_return": episode_reward, "test_return_ma": sim_params.test.test_return_ma.get_average(), "test_num": sim_params.test.test_num})


def main():
    sim_params = get_params()
    sim_params = DDPG_setup(sim_params)
        
    num_workers = sim_params.ddpg_args.n_workers

    env = create_test_env(sim_params)
    
    main_agent = DDPGagent(sim_params.ddpg_args.use_cuda, 
                           sim_params.ddpg_args.n_states,
                           sim_params.ddpg_args.n_actions,
                           hidden_size= sim_params.ddpg_args.mlp_hidden_dim,
                           actor_learning_rate= sim_params.ddpg_args.lr,
                           critic_learning_rate= sim_params.ddpg_args.critic_lr,
                           tau= sim_params.ddpg_args.tau,
                           max_memory_size= sim_params.ddpg_args.max_memory_size,
                           directory= sim_params.ddpg_args.weights_dir) 

    if sim_params.load:
        main_agent.load()
    
    start_training_time=datetime.now()

    vec_envs = VecEnv(num_workers, sim_params)    

    episode_reward_best = -300
    #Debugging variables
    firstEpi = True
    firstRet = 0 
    finalRet = 0
    notTraining = True

    for episode in range(sim_params.n_episodes): 
        
        states, panda_pos_orien = vec_envs.reset()
        episode_reward =  np.zeros(num_workers)

        for step in range(sim_params.n_steps):
            action = main_agent.get_action(states)
            next_state, rewards, done, info, panda_pos_orien = vec_envs.step(action)
            for i in range(num_workers):
                main_agent.memory.push(states[i], action[i], rewards[i], next_state[i], done[i]) 

            if len(main_agent.memory) > sim_params.batch_size:
                if notTraining:
                    print("Started Training .......................... ")
                    notTraining = False
                main_agent.update(sim_params.batch_size)

            states = next_state
            episode_reward += rewards

            if step % 10 == 0:
                print('step: {}'.format(step))

            if step % 100 == 0:
                for i in range(num_workers):
                    data_info_debug = '=> [{}] worker_id: {}, reward: {:.4f}, episode: {}, step: {}, done: {}, accum_ret: {:.4f}, action{}\n'.format(datetime.now(), i+1, rewards[i], episode+1, step, done[i], episode_reward[i],np.round(action[i].flatten(),4))
                    print(data_info_debug)

        max_reward = np.max(episode_reward)
        if max_reward > episode_reward_best:
            episode_reward_best = max_reward
            main_agent.save()

        if firstEpi:
            firstRet = episode_reward_best
            firstEpi = False

        for i in range(num_workers):
            vec_envs.moving_avg_return[i].add_reward(episode_reward[i])
            reward_data = '=> [{}] worker: {}, return: {:.4f}, episode: {}/{}, return_ma: {:.4f}'.format(datetime.now(), i+1, episode_reward[i], episode+1, sim_params.n_episodes, vec_envs.moving_avg_return[i].get_average())
            print(reward_data)
            worker_id_str = str(i+1)
            return_str = "return_"+worker_id_str
            return_ma_str = "return_ma_"+worker_id_str
            wandb.log({return_str: episode_reward[i], return_ma_str:  vec_envs.moving_avg_return[i].get_average()})

        wandb.log({"episode": episode + 1})
        if episode % 10 == 0:
            test_env(env, main_agent, sim_params)
            main_agent.save()

        update_msg = '=> [{}] Episode finished:  episode: {}/{}'.format(datetime.now(),  episode+1, sim_params.n_episodes)
        print(update_msg)
        
    finalRet = episode_reward_best
    vec_envs.close()
  
    main_agent.save()
    print("end mode train !")
    print("time elapsed = {}".format(datetime.now()-start_training_time))

    sim_params.wandb_run.alert(
    title="Training finished",
    text=f"First return {firstRet}, final return {finalRet}",
    level=AlertLevel.INFO,
    )
    wandb.finish()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

