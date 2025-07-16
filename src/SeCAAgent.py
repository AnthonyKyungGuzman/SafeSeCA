from SeCA.learners.seca_learner import SeCALearner
from SeCA.controllers.basic_controller import BasicMAC
from SeCA.learners import REGISTRY as le_REGISTRY
from SeCA.controllers import REGISTRY as mac_REGISTRY
from utils import sync_networks, Memory
import torch
from functools import partial
import numpy as np

class SeCAAgent:
    def __init__(self, args,env):

        self.args = args
        
		# Setup multiagent controller here
        self.mac = mac_REGISTRY[args.mac](self.args.scheme, args.groups, args)

        # Learner
        self.learner = le_REGISTRY[args.learner](self.mac, self.args.scheme, args.logger, args)

        if args.use_cuda:
            self.learner.cuda()

        self.env = env
        self.t_env = -1 # counter for logging


        self.memory = Memory(args.max_memory_size)   


    def load(self): 
        print("Loading weights ...")
        self.learner.load_models(self.args.weights_dir)


    def update(self,batch_size, episode):
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(np.array(states_batch))
        actions = torch.FloatTensor(np.array(actions_batch))
        rewards = torch.FloatTensor(np.array(rewards_batch))
        next_states = torch.FloatTensor(np.array(next_states_batch))
        dones = torch.from_numpy(np.vstack(dones_batch)).float()
        
        if self.args.use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()
            dones = dones.cuda()

        batch = [states, actions, rewards, next_states, dones]
    
        self.learner.train_critic(batch, self.t_env, episode)
        self.learner.train(batch, self.t_env, episode, self.env)


    def save(self, best=False):
        self.learner.save_models(self.args.weights_dir,best=best)

    def get_action(self,states, step, panda_pos_orien, test_mode=False):
        states = torch.from_numpy(states).float().unsqueeze(0)
        if self.args.use_cuda:
           states = states.cuda()
        if not test_mode:
           states.squeeze_(0)

        if self.t_env*states.shape[0] < self.args.batch_size and test_mode == False:
            actions = 2 * np.random.rand(states.shape[0], 2, 3) - 1
            actions_nn = actions
        else:
            with torch.no_grad():
                actions, actions_nn = self.mac.select_actions(states, self.env, panda_pos_orien, step, test_mode=test_mode)
        self.t_env += 1
        if not test_mode:
            return actions, actions_nn
        else:
            return actions

    def get_count_of_actions(self):
        return self.mac.get_count_of_actions()