import torch.nn as nn
import torch.nn.functional as F
import torch as th


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        if args.nn_type == "RNN":
            input_shape -= 2*args.n_actions #removed mean and std for rotation agent
            translation_input_last_layer = args.rnn_hidden_dim+2*args.n_actions
            output_shape = 2*args.n_actions

            self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
            self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

            self.fc_mean_std_r = nn.Linear(args.rnn_hidden_dim, output_shape)
            self.fc_mean_std_t = nn.Linear(translation_input_last_layer, output_shape)

        elif args.nn_type == "MLP":

            self.fc1 = nn.Sequential(nn.Linear(input_shape, args.mlp_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim))
            self.fc_mean = nn.Linear(args.mlp_hidden_dim, args.n_actions)
            self.fc_log_std = nn.Linear(args.mlp_hidden_dim, args.n_actions)

        elif args.nn_type == "MLP2":

            input_shape -= 2*args.n_actions #removed mean and std for rotation agent

            translation_input_last_layer = args.mlp_hidden_dim+2*args.n_actions
            output_shape = 2*args.n_actions

            self.fc1 = nn.Sequential(nn.Linear(input_shape, args.mlp_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim))
            self.fc_mean_std_r = nn.Linear(args.mlp_hidden_dim, output_shape)
            # self.fc_log_std_r = nn.Linear(args.mlp_hidden_dim, args.n_actions)

            self.fc_mean_std_t = nn.Linear(translation_input_last_layer, output_shape)
            # self.fc_log_std_t = nn.Linear(args.mlp_hidden_dim, args.n_actions)


    def init_hidden(self):
        # make hidden states on same device as model
        if self.args.nn_type == "RNN":
            return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        else:
            return self.fc1[0].weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):

        if self.args.nn_type == "RNN":
            x = F.relu(self.fc1(inputs))
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(x, h_in)

            mean_log_std_1 = self.fc_mean_std_r(h)

            mean_1 = mean_log_std_1[:,:self.args.n_actions]
            log_std_1 = mean_log_std_1[:,self.args.n_actions:]
   
            std = th.clamp(log_std_1, min=-20, max=2).exp()

            inputs_a2 = th.cat([h, mean_1, std], dim=-1)    
            mean_log_std_2 = self.fc_mean_std_t(inputs_a2)

            mean_2 = mean_log_std_2[:,:self.args.n_actions]
            log_std_2 = mean_log_std_2[:,self.args.n_actions:]

            mean = th.stack([mean_1, mean_2], dim=1)   
            log_std = th.stack([log_std_1, log_std_2], dim=1)  
            return mean,log_std, h
        elif self.args.nn_type == "MLP":
            zeros = th.zeros(inputs.shape[0], 2*self.args.n_actions, device=self.args.device)
            inputs_a1 = th.cat([inputs, zeros], dim=-1) 

            x = F.relu(self.fc1(inputs_a1))
            mean_1 = self.fc_mean(x)
            log_std_1= self.fc_log_std(x)

            std = th.clamp(log_std_1, min=-20, max=2).exp()

            inputs_a2 = th.cat([inputs, mean_1, std], dim=-1)    

            x = F.relu(self.fc1(inputs_a2))
            mean = self.fc_mean(x)
            log_std = self.fc_log_std(x)
           
            mean = th.stack([mean_1, mean], dim=1)   
            log_std = th.stack([log_std_1, log_std], dim=1)  
            
            return mean, log_std, None
        
        elif self.args.nn_type == "MLP2":

            inputs_a1 = inputs

            x = F.relu(self.fc1(inputs_a1))
            mean_log_std_1 = self.fc_mean_std_r(x)

            mean_1 = mean_log_std_1[:,:self.args.n_actions]
            log_std_1 = mean_log_std_1[:,self.args.n_actions:]
   
            std = th.clamp(log_std_1, min=-20, max=2).exp()

            inputs_a2 = th.cat([x, mean_1, std], dim=-1)    
            mean_log_std_2 = self.fc_mean_std_t(inputs_a2)

            mean_2 = mean_log_std_2[:,:self.args.n_actions]
            log_std_2 = mean_log_std_2[:,self.args.n_actions:]

            mean = th.stack([mean_1, mean_2], dim=1)   
            log_std = th.stack([log_std_1, log_std_2], dim=1)  
            
            return mean, log_std, None
        

    def sample(self,state, hidden_states, reparametrized=False):
        if self.args.nn_type == "RNN":
            mean, log_std, h = self(state,hidden_states)
        else:
            mean, log_std, h = self(state,None)
        std = th.clamp(log_std, min=-20, max=2).exp()
        # dist = th.distributions.Normal(mean, std)
        cov_matrix = th.diag_embed(std ** 2) 
        dist = th.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov_matrix)
        if reparametrized:
            z = dist.rsample() + 1e-6
        else:
            z = dist.sample() + 1e-6
        action = th.tanh(z)
        log_prob = dist.log_prob(z) - th.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        # log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy()
        return action, log_prob, entropy, mean, std, h
    
    def sample_wo_action_bounds(self,state, hidden_states, reparametrized=False):
        if self.args.nn_type == "RNN":
            mean, log_std, h = self(state,hidden_states)
        else:
            mean, log_std, h = self(state,None)
        std = th.clamp(log_std, min=-20, max=2).exp()
        # dist = th.distributions.Normal(mean, std)
        cov_matrix = th.diag_embed(std ** 2) 
        dist = th.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov_matrix)
        if reparametrized:
            z = dist.rsample()
        else:
            z = dist.sample()
        action = z
        log_prob = dist.log_prob(z)
        entropy = dist.entropy()
        return action, log_prob, entropy, mean, std, h
    
   


