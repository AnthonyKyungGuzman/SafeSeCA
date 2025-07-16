from SeCA.modules.agents import REGISTRY as agent_REGISTRY
import torch as th
import numpy as np
from Action_Masking.action_masking import ActionMasking
import copy


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)

        self.hidden_states = []

        self.total_action_count = 0
        self.unsafe_action_count = 0
        self.safe_action = 1

        
        ########
        self.reset_actions()
        self.noise = args.noise
        #Action masking
        if self.args.action_masking:
            self.masking = ActionMasking(args) #using same action masking params for both agents

        


    def select_actions(self, states, env, pos_orients, step, bs=slice(None), test_mode=False):
        agent_mean, agent_log_std, self.hidden_states = self.agent(states, self.hidden_states)
        agent_std = th.clamp(agent_log_std, min=-20, max=2).exp()
        cov_matrix = th.diag_embed(agent_std ** 2)  # 

        if len(agent_mean.shape) > 2:
            dists = [th.distributions.MultivariateNormal(loc=agent_mean[i],covariance_matrix=cov_matrix[i]) for i in range(agent_mean.shape[0])]
            viable_arr = np.zeros(agent_mean.shape[0])
        else:
            dists = th.distributions.MultivariateNormal(loc=agent_mean, covariance_matrix=cov_matrix)

        viable = viable_translation = viable_orientation = False
        action = None
        action_np = None


        while not viable or not viable_translation or not viable_orientation:
            if len(agent_mean.shape) > 2:
                if np.sum(viable_arr == 0) == agent_mean.shape[0]:
                    # if test_mode:
                    #     action = agent_mean
                    # else:
                    action = th.stack([dist.sample() for dist in dists]) 
                    condition = (action > self.safe_action) | (action < -self.safe_action)
                    if condition.any():
                        self.unsafe_action_count += 1
                    self.total_action_count +=1 
                    action = th.tanh(action)
                    action_nn_np = copy.copy(action)
                    action_nn_np = action_nn_np.squeeze(0).cpu().detach().numpy() if self.args.use_cuda else action_nn_np.squeeze(0).detach().numpy()
                    if self.args.action_masking:
                        action = self.masking.ray_masking(action)
                else:
                    zero_indices = np.where(viable_arr == 0)[0]
                    action[zero_indices] = th.stack([dists[i].sample() for i in zero_indices])
                    action[zero_indices]  = th.tanh(action[zero_indices])
                    action_nn_np = copy.copy(action)
                    action_nn_np = action_nn_np.squeeze(0).cpu().detach().numpy() if self.args.use_cuda else action_nn_np.squeeze(0).detach().numpy()
                    if self.args.action_masking:
                        action[zero_indices] = self.masking.ray_masking(action[zero_indices])
            else:
                # if test_mode:
                #     action = agent_mean
                # else:
                action = dists.sample() 
                action = th.tanh(action)
                action_nn_np = copy.copy(action)
                action_nn_np = action_nn_np.squeeze(0).cpu().detach().numpy() if self.args.use_cuda else action_nn_np.squeeze(0).detach().numpy()
                if self.args.action_masking:
                    action = self.masking.ray_masking(action)

            action_np = action.squeeze(0).cpu().detach().numpy() if self.args.use_cuda else action.squeeze(0).detach().numpy()

            if self.args.check_viability:
                if  len(action_np.shape) > 2:
                    if np.sum(viable_arr == 0) == agent_mean.shape[0]:
                        for i in range(action.shape[0]):
                            action_np_check = action_np[i]
                            pos_orien = pos_orients[i]
                            env.go_to_cartesian_bullet(pos_orien[0], pos_orien[1], 0)
                            viable_translation = env.compute_translation_viability_of_action(action_np_check[0],0) # 0 is the panda number (currently only using 1 gripper)
                            viable_orientation = env.compute_orientation_viability_of_action(action_np_check[1],0) # 0 is the panda number (currently only using 1 gripper)
                            viable = env.compute_translation_orientation_viability_of_action(action_np_check[0],action_np_check[1],0) # 0 is the panda number (currently only using 1 gripper)
                            viable_arr[i] = int(viable_translation and viable_orientation and viable)
                    else:
                        zero_indices = np.where(viable_arr == 0)[0]
                        for i in zero_indices:
                            action_np_check = action_np[i]
                            pos_orien = pos_orients[i]
                            env.go_to_cartesian_bullet(pos_orien[0], pos_orien[1], 0)
                            viable_translation = env.compute_translation_viability_of_action(action_np_check[0],0) # 0 is the panda number (currently only using 1 gripper)
                            viable_orientation = env.compute_orientation_viability_of_action(action_np_check[1],0) # 0 is the panda number (currently only using 1 gripper)
                            viable = env.compute_translation_orientation_viability_of_action(action_np_check[0],action_np_check[1],0) # 0 is the panda number (currently only using 1 gripper)
                            viable_arr[i] = int(viable_translation and viable_orientation and viable)
                    if np.sum(viable_arr == 0) >= 1: 
                        viable = False
                else:
                    viable_translation = env.compute_translation_viability_of_action(action_np[0],0) # 0 is the panda number (currently only using 1 gripper)
                    viable_orientation = env.compute_orientation_viability_of_action(action_np[1],0) # 0 is the panda number (currently only using 1 gripper)
                    viable = env.compute_translation_orientation_viability_of_action(action_np[0],action_np[1],0) # 0 is the panda number (currently only using 1 gripper)
            else:
                break

        return action_np, action_nn_np
    

    
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, 1, -1)  # bav

        return self.hidden_states

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path, best=False):
        agent_str = self.get_strings_for_load_and_save(best=best)
        th.save(self.agent.state_dict(), agent_str.format(path))


    def get_strings_for_load_and_save(self,best=False):
        prefix_str = "{}/"
        extension_str = ".th"
        agent_str = prefix_str + "agent"

        if best:
            agent_str += "_best" 
        
        agent_str +=  extension_str
        return agent_str

    def load_models(self, path):
        agent_str = self.get_strings_for_load_and_save(best=self.args.load_best)
        self.agent.load_state_dict(th.load(agent_str.format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape): #TODO: make n agents networks
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape
    
    def reset_actions(self):
        self.agents_prev_actions = th.zeros(self.n_agents, self.args.n_actions*2, device="cuda" if self.args.use_cuda else "cpu") #*2 beacuse it is mean and std

    def get_count_of_actions(self):
        return self.unsafe_action_count, self.total_action_count

  