from SeCA.learners.seca_learner import SeCALearner
from SeCA.controllers.basic_controller import BasicMAC
from SeCA.learners import REGISTRY as le_REGISTRY
from SeCA.controllers import REGISTRY as mac_REGISTRY
from SeCA.components.episode_buffer import ReplayBuffer, EpisodeBatch
from utils import sync_networks, Memory
import torch
from functools import partial

class SeCA_agent_envs:
    def __init__(self, args,env):

        self.args = args

		# Setup multiagent controller here
        self.mac = mac_REGISTRY[args.mac](self.args.scheme, args.groups, args)

        if args.use_cuda:
            self.mac.cuda()

        self.env = env
        self.t_env = -1 # counter for logging
        self.episode_num = 0
                

    def get_action(self,states, step, test_mode=False):
        states = torch.from_numpy(states).float().unsqueeze(0)
        if self.args.use_cuda:
           states = states.cuda()
        actions, mean, std = self.mac.select_actions(states, self.env, step, test_mode=test_mode)
        self.t_env += 1
        return actions, mean, std

        # self.mac.select_actions(states)