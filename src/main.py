"""
Laurent LEQUIEVRE
Research Engineer, CNRS (France)
ISPR - MACCS Team
Institut Pascal UMR6602
laurent.lequievre@uca.fr

Melodie DANIEL
Post-doctorate
ISPR - MACCS Team
Institut Pascal UMR6602
melodie.daniel@sigma-clermont.fr
"""



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

from mpi4py import MPI
import pybullet as p

import gym_panda_frite
from database_frite import Database_Frite

from gym_panda_frite.envs.environment import Environment

from gym_panda_frite.envs.panda_frite_env_wrapper_reward_mean import PandaFriteEnvWrapperRewardMean
from gym_panda_frite.envs.panda_frite_env_wrapper_reward_max import PandaFriteEnvWrapperRewardMax
from gym_panda_frite.envs.panda_frite_env_wrapper_reward_dtw import PandaFriteEnvWrapperRewardDTW

####################### Added ################################
from SeCAAgent import SeCAAgent
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
##############################################################

from json_decoder import JsonDecoder

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='test', type=str) # mode = 'train' or 'test' or 'test_real' or 'generate_db'
parser.add_argument('--gui', default=False, type=bool) # use gui to see graphic results
parser.add_argument('--config_file', default='./configs/default/default.json', type=str)
parser.add_argument('--algo_config_file', default='./SeCA/config/algs/seca.yaml', type=str)

args = parser.parse_args()

wandb.login()
wandb.tensorboard.patch()

def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    return config

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def _get_algo_config(config_name, subfolder):
    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "SeCA/config/", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f,Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict
	


def main():
	
	if not os.path.isfile(args.config_file):
			raise RuntimeError("=> Config file JSON to load does not exit : " + args.config_file)
			return

	json_decoder = JsonDecoder(args.config_file)


	os.environ['OMP_NUM_THREADS'] = '1'
	os.environ['MKL_NUM_THREADS'] = '1'
	os.environ['IN_MPI'] = '1'
	
	env_name = json_decoder.config_data["env"]["name"]
	env_random_seed = json_decoder.config_data["env"]["random_seed"]
	env_time_set_action = json_decoder.config_data["env"]["time_set_action"]
	env_reward_type = json_decoder.config_data["env"]["reward_parameters"]["reward_index"]
	
	print("** ENV PARAMETERS **")
	print("env_name = {}".format(env_name))
	print("env_random_seed = {}".format(env_random_seed))
	print("env_time_set_action = {}".format(env_time_set_action))
	
	ddpg_cuda = json_decoder.config_data["ddpg"]["cuda"]
	ddpg_max_memory_size = json_decoder.config_data["ddpg"]["max_memory_size"]
	ddpg_batch_size = json_decoder.config_data["ddpg"]["batch_size"]
	ddpg_log_interval = json_decoder.config_data["ddpg"]["log_interval"]
	
	print("** DDPG PARAMETERS **")
	print("ddpg_cuda = {}".format(ddpg_cuda))
	print("ddpg_batch_size = {}".format(ddpg_batch_size))
	print("ddpg_max_memory_size = {}".format(ddpg_max_memory_size))
	print("ddpg_log_interval = {}".format(ddpg_log_interval))
	
	log_name = json_decoder.config_data["log"]["name"]
	database_name = json_decoder.config_data["database"]["name"]
	
	print("** LOG PARAMETERS **")
	print("log_name = {}".format(log_name))
	
	rank = MPI.COMM_WORLD.Get_rank()
	size = MPI.COMM_WORLD.Get_size() 

	########################## SeCA Params
	# Get the defaults from default.yaml
	with open(os.path.join(os.path.dirname(__file__), "SeCA/config", "default.yaml"), "r") as f:
		try:
			seca_config_dict = yaml.load(f,Loader=yaml.FullLoader)
		except yaml.YAMLError as exc:
			assert False, "default.yaml error: {}".format(exc)

	alg_config = _get_algo_config(args.algo_config_file, "algs")
	seca_config_dict = recursive_dict_update(seca_config_dict, alg_config)
	
	_config = args_sanity_check(seca_config_dict,_logger)
	# Save to disk by default for sacred
	_logger.info("SeCA Config initiated")

	seca_args = SN(**_config)
	seca_args.device = "cuda" if seca_args.use_cuda else "cpu"

	# setup loggers
	logger = Logger(_logger)


	# configure tensorboard logger
	unique_token = "{}__{}__{}".format(seca_args.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), rank)
	seca_args.unique_token = unique_token
	if seca_args.use_tensorboard:
		# tb_logs_direc = "/kpl/tensorboard"
		tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "src/SeCA/results", "tb_logs")
		tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)

		tb_exp_direc_path = Path(tb_exp_direc)
		tb_exp_direc_path.mkdir(exist_ok=True)

		os.environ["TENSORBOARD_LOGDIR"] = tb_exp_direc
		wandb_run = wandb.init(project="seca_dlo",
							name=f"rank_{rank}",
							group="test",
							config={"rank": rank, "total_processes": size},
							sync_tensorboard=True,
							dir=tb_exp_direc  # <- this makes it easier for wandb to find your logs
						)
		log_dir = os.path.join("runs", wandb.run.id)
		print(f"[DEBUG] Logging to {log_dir}")

		logger.setup_tb(log_dir)
		# logger.setup_tb(tb_exp_direc)
		logger.tb_logger.add_scalar("init_log_tb",0,0)
		logger.tb_logger.flush()
	

	#######################################

				
	if not os.path.isfile(json_decoder.config_dir_name  + database_name):
		raise RuntimeError("=> Database file to load does not exit : " + json_decoder.config_dir_name  + database_name)
		return

	configsParentDir = Path(json_decoder.config_dir_name)
	logsFileDir = str(configsParentDir.parent) + "/logs/"

	ddpg_load = json_decoder.config_data["ddpg"]["load"]
	load_from_dir = json_decoder.config_data["ddpg"]["load_from_dir"]
	weights_folder = json_decoder.config_data["ddpg"]["weights_folder"] 
	
	weights_dir = ""

	# #select latest weights to load to nets
	if rank==0:
		if ddpg_load:
			directory = str(configsParentDir.parent) + "/SeCA/weights/" 
			if load_from_dir:
				weights_dir = directory + weights_folder
			else:
				folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
				weights_dir = max(folders, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
				weights_dir = str(configsParentDir.parent) + "/SeCA/weights/" + weights_dir
		else: #create new folder for weights
			weights_dir = str(configsParentDir.parent) + "/SeCA/weights/" + "{}__{}".format(seca_args.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) 
			weights_dir = Path(weights_dir)
			weights_dir.mkdir(exist_ok=True)

	# weights_dir =	"/home/anthony/research/DLO/MultiSeCA_DLO/SeCA/weights/seca__2025-04-09_19-41-13/"

	envLogFile = open(str(configsParentDir.parent) + "/logs/"	 + json_decoder.config_data["log"]["env_name"], "w+")

	if rank == 0:
		file_log = open(logsFileDir + log_name, "w+")
		
	env_pybullet = Environment(json_decoder=json_decoder, gui=args.gui)
	env_pybullet.startThread()
	
	db = Database_Frite(json_decoder=json_decoder)
	
	# what kind of reward ? ( 0 = Mean, 1 = Max, 2 = DTW)
	if env_reward_type == 0:
			env = PandaFriteEnvWrapperRewardMean(gym.make(env_name, database=db, json_decoder = json_decoder, env_pybullet=env_pybullet, gui=args.gui, env_rank=rank))
	elif env_reward_type == 1:
			env = PandaFriteEnvWrapperRewardMax(gym.make(env_name, database=db, json_decoder = json_decoder, env_pybullet=env_pybullet, gui=args.gui, env_rank=rank))
	
	elif env_reward_type == 2:
			env = PandaFriteEnvWrapperRewardDTW(gym.make(env_name, database=db, json_decoder = json_decoder, env_pybullet=env_pybullet, gui=args.gui, env_rank=rank))
	else:
			raise RuntimeError("=> NO REWARD TYPE SPECIFIED INTO CONFIG FILE !!!!")
			return
	
	#env = gym.make(env_name, database=db, json_decoder = json_decoder, env_pybullet=env_pybullet, gui=args.gui, env_rank=rank)
	#env = PandaFriteEnvWrapperRewardMax(gym.make(env_name, database=db, json_decoder = json_decoder, env_pybullet=env_pybullet, gui=args.gui, env_rank=rank))
	#env = PandaFriteEnvWrapperRewardMean(gym.make(env_name, database=db, json_decoder = json_decoder, env_pybullet=env_pybullet, gui=args.gui, env_rank=rank))
	
	env.seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
	torch.manual_seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
	#np.random.seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
    
	if (seca_args.use_cuda):
		torch.cuda.manual_seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
	
	#Params for SeCA Agent
	seca_args.n_actions = env.get_nb_action_values()
	seca_args.n_states = env.get_nb_obs_values()
	seca_args.max_memory_size = ddpg_max_memory_size
	seca_args.weights_dir = str(weights_dir)
	seca_args.batch_size = ddpg_batch_size
	seca_args.logger = logger
	seca_args.state_shape = seca_args.n_states - 6 # removed 6, so the states going into the criitc do not include any action distribution params, which will be added later at the time of inference

	 # Default/Base scheme
	scheme = {
		"state": {"vshape": seca_args.state_shape }, 
		"obs": {"vshape": seca_args.n_states}, 
		# "obs": {"vshape": seca_args.n_states, "group": "agents"}, 
		"actions": {"vshape": (seca_args.n_actions,), "group": "agents", "dtype": th.float32},
		"avail_actions": {"vshape": (seca_args.n_actions,), "group": "agents", "dtype": th.float32},
		"reward": {"vshape": (1,)},
		"terminated": {"vshape": (1,), "dtype": th.bool},
		"actions_mean":{"vshape": (seca_args.n_actions,), "group": "agents", "dtype": th.float32},
		"actions_std":{"vshape": (seca_args.n_actions,), "group": "agents", "dtype": th.float32},
	}

	groups = {
        "agents": seca_args.n_agents
    }
	seca_args.scheme = scheme
	seca_args.groups = groups
	seca_args.episode_steps = json_decoder.config_data["env_train"]["n_steps"] #TODO: change steps when testing
	seca_args.noise = OUNoiseTorch(seca_args.n_agents, env.get_nb_action_values(), env.get_action_space_low() , env.get_action_space_high(), device=seca_args.device)

	# agent = DDPGagent(ddpg_cuda, num_states=env.get_nb_obs_values(), num_actions=env.get_nb_action_values(), max_memory_size=ddpg_max_memory_size, directory=json_decoder.config_dir_name+'env_panda/')
	agent = SeCAAgent(seca_args, env)
    
	list_global_rewards = []

	panda_num = 0 #TODO: change to number of panda 
    
	if args.mode == 'test':
		print("TEST MODE !")
		
		start=datetime.now()
		
		file_log.write("mode test !\n")
		
		agent.load()
		n_episodes = json_decoder.config_data["env_test"]["n_episodes"]
		n_steps = json_decoder.config_data["env_test"]["n_steps"]
		do_reset_env = json_decoder.config_data["env"]["do_reset_env"]
		wait_time_sleep_after_draw_env_box = json_decoder.config_data["env_test"]["wait_time_sleep_after_draw_env_box"]
		wait_time_sleep_end_episode = json_decoder.config_data["env_test"]["wait_time_sleep_end_episode"]
		do_episode_hit_return = json_decoder.config_data["env_test"]["do_episode_hit_return"]
		
		file_log.write("** ENV MODE TEST **\n")
		file_log.write("config_file = {}\n".format(args.config_file))
		file_log.write("n_episodes = {}\n".format(n_episodes))
		file_log.write("n_steps = {}\n".format(n_steps))
		file_log.write("do_reset_env = {}\n".format(do_reset_env))
		file_log.write("wait_time_sleep_after_draw_env_box = {}\n".format(wait_time_sleep_after_draw_env_box))
		file_log.write("wait_time_sleep_end_episode = {}\n".format(wait_time_sleep_end_episode))
		file_log.write("do_episode_hit_return = {}\n".format(do_episode_hit_return))
		
		print("** ENV MODE TEST **")
		print("n_episodes = {}".format(n_episodes))
		print("n_steps = {}".format(n_steps))
		print("do_reset_env = {}".format(do_reset_env))
		print("wait_time_sleep_after_draw_env_box = {}".format(wait_time_sleep_after_draw_env_box))
		print("wait_time_sleep_end_episode = {}".format(wait_time_sleep_end_episode))
		print("do_episode_hit_return = {}".format(do_episode_hit_return))
				
		nb_dones = 0
		sum_distance_error = 0
		list_episode_error = []
		sum_steps = 0
		
		# initial error
		initial_error = 0
		sum_initial_error = 0
		list_initial_error = []
		
		

		for episode in range(n_episodes):
			print("Episode : {}".format(episode))
			file_log.write("Episode : {}\n".format(episode))
			file_log.flush()
		   
			if do_reset_env:
				print("RESET ENV !")
				file_log.write("RESET ENV !\n")
				file_log.flush()
				env.reset_env_bullet()
							
			state, initial_error = env.reset_bullet(panda_num)
			agent.mac.init_hidden(batch_size=1)
			agent.mac.reset_actions()
			file_log.write("initial distance error = {}\n".format(initial_error))
			file_log.flush()
			list_initial_error.append(initial_error)
			sum_initial_error+=initial_error
				
			# if (args.gui):
			# 	env.draw_env_box()
			# 	env.draw_frite_parameters()
			# 	time.sleep(wait_time_sleep_after_draw_env_box)
			   
			current_distance_error = 0
			if (args.gui) and env.is_graphic_mode == False:
					env.draw_id_to_follow()
			
			for step in range(n_steps):
				# action = agent.get_action(state)
				pos, orien = env.get_gripper_pos_orientation(panda_num)
				pos_orien = [pos, orien]
				action = agent.get_action(state, step, pos_orien,test_mode=True)
				
				print("action={}".format(action))
				# file_log.write("action = {}\n".format(action))
				# file_log.flush()

				
				new_state, reward, done, info = env.step_bullet(action, panda_num, rank=rank, episode=episode, step=step)
				current_distance_error = info['distance_error']
				if (args.gui) and env.is_graphic_mode == False:
					env.draw_id_to_follow()
				
				print("step={}, distance_error={}\n".format(step,info['distance_error']))
				file_log.write("step={}, distance_error={}\n".format(step,info['distance_error']))
				file_log.flush()
				
				#print("step={}, action={}, reward={}, done={}, info={}".format(step,action,reward, done, info))
				state = new_state
			   
				if done:
					print("done with step={}  !".format(step))
					file_log.write("done with step={}  !\n".format(step))
				   
					nb_dones+=1
					break
				   
			if (args.gui):
				if do_episode_hit_return:
					input("hit return to continue !")
				else:
					time.sleep(wait_time_sleep_end_episode)
			
			sum_steps += step
			sum_distance_error += current_distance_error
			list_episode_error.append(current_distance_error)
		print("time_set_action = {}".format(env_time_set_action))
		print("nb dones = {}".format(nb_dones))
		print("mean distance error = {}".format(sum_distance_error/n_episodes))
		print("sum distance error = {}".format(sum_distance_error))
		print("Average steps =  {}".format(sum_steps/n_episodes))
		print("std error episode = {}, min error episode = {}, max error episode = {}".format(np.std(np.array(list_episode_error)),min(list_episode_error),max(list_episode_error)))
		print("success rate = {}".format(nb_dones/n_episodes))
		# initial error
		print("mean initial distance error = {}".format(sum_initial_error/n_episodes))
		print("max initial distance error = {}".format(max(list_initial_error)))
		print("min initial distance error = {}".format(min(list_initial_error)))
		print("time elapsed = {}".format(datetime.now()-start))
		
		
		file_log.write("time_set_action = {}\n".format(env_time_set_action))
		file_log.write("nb dones = {}\n".format(nb_dones))
		file_log.write("mean distance error = {}\n".format(sum_distance_error/n_episodes))
		file_log.write("sum distance error = {}\n".format(sum_distance_error))
		file_log.write("Average steps =  {}\n".format(sum_steps/n_episodes))
		file_log.write("std error episode = {}, min error episode = {}, max error episode = {}\n".format(np.std(np.array(list_episode_error)),min(list_episode_error),max(list_episode_error)))
		file_log.write("success rate = {}\n".format(nb_dones/n_episodes))
		# initial error
		file_log.write("mean initial distance error = {}\n".format(sum_initial_error/n_episodes))
		file_log.write("max initial distance error = {}\n".format(max(list_initial_error)))
		file_log.write("min initial distance error = {}\n".format(min(list_initial_error)))
		file_log.write("time elapsed = {}\n".format(datetime.now()-start))
		file_log.flush()
		file_log.close()
		
		# input("hit return to exit !")
		
	elif args.mode == 'generate_db':
		print("GENERATE DB MODE !")
		db.print_config_db()
		input("hit return to start generate db !")
		db.generate()
		# input("hit return to exit !")
				
	else:
		raise NameError("mode wrong!!!")
	
	wandb.finish()
   
        
if __name__ == '__main__':
	main()


