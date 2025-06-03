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
import numpy as np
import random
import torch
import os

import matplotlib.pyplot as plt
from utils import *

import argparse
import time
from datetime import datetime

from mpi4py import MPI
import pybullet as p

from gym_panda_frite.envs.environment import Environment

from json_decoder import JsonDecoder
from pathlib import Path

from database_frite import Database_Frite

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test' or 'test_real' or 'generate_db'
parser.add_argument('--gui', default=False, type=bool) # use gui to see graphic results
parser.add_argument('--config_file', default='./configs/default/default.json', type=str)

args = parser.parse_args()

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
	
	print("** ENV PARAMETERS **")
	print("env_name = {}".format(env_name))
	print("env_random_seed = {}".format(env_random_seed))
	print("env_time_set_action = {}".format(env_time_set_action))
	
	rank = MPI.COMM_WORLD.Get_rank()

	configsParentDir = Path(json_decoder.config_dir_name)
	envLogFile = open(str(configsParentDir.parent) + "/logs/"	 + json_decoder.config_data["log"]["env_name"], "w+")
		
	env_pybullet = Environment(json_decoder=json_decoder, gui=args.gui)
	env_pybullet.startThread()

	db = Database_Frite(json_decoder=json_decoder)
	env = gym.make(env_name, database=db, json_decoder = json_decoder, env_pybullet=env_pybullet, gui=True, env_rank=rank)
	
	torch.manual_seed(env_random_seed + MPI.COMM_WORLD.Get_rank())

	while (True):
		time.sleep(1)


   
        
if __name__ == '__main__':
	main()


