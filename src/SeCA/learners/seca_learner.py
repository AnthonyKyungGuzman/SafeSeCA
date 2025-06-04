import copy
from SeCA.modules.critics.seca import SeCACritic
import torch as th
from torch.optim import RMSprop, Adam
import math

from datetime import datetime

#action masking
from Action_Masking.action_masking import ActionMasking


class SeCALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        del args.logger

        self.last_target_update_episode = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.log_stats_t_agent = -self.args.learner_log_interval - 1

        self.critic_q1 = SeCACritic(scheme, args)
        self.target_critic_q1 = copy.deepcopy(self.critic_q1)

        self.critic_q2 = SeCACritic(scheme, args)
        self.target_critic_q2 = copy.deepcopy(self.critic_q2)

        self.agent_params = list(self.mac.parameters())
        self.critic_params_q1 = list(self.critic_q1.parameters())
        self.critic_params_q2 = list(self.critic_q2.parameters())
        self.params = self.agent_params + self.critic_params_q1 + self.critic_params_q2

        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)
        self.critic_q1_optimiser = Adam(params=self.critic_params_q1, lr=args.critic_lr)
        self.critic_q2_optimiser = Adam(params=self.critic_params_q2, lr=args.critic_lr)

        self.entropy_coef = args.entropy_coef

        self.init_order()
        if self.args.action_masking:
            self.masking = self.mac.masking
   
    def calc_F(self, actions, states):
        q1_pred = self.critic_q1(actions, states)
        q2_pred = self.critic_q2(actions, states)
            
        F = th.min(q1_pred, q2_pred)
        return F 
    
    def calc_advantage(self, states, means, reparam_actions):
        
        # Calculate adv with F2 - F1
        adv = [None] * self.args.n_agents
        F1 = self.calc_F(means, states)
        for i in self.order:
            tmp = reparam_actions.clone() 
            if i < self.args.n_agents - 1:
                tmp[:, i+1:, :] = means[:, i+1:, :]
            F2 = self.calc_F(tmp, states)
            if i > 0:
                tmp = reparam_actions.clone() 
                tmp[:, i:, :] = means[:, i:, :]
                # Call critic
                F1 = self.calc_F(tmp, states)
            adv[i] = (F2-F1).clone().detach()
        adv = th.cat(adv, dim=-1).squeeze(dim=1) 
        
        return adv


    def train(self, batch, t_env: int, episode_num: int, env):

        states, actions, rewards, next_states, dones = batch  

        reparam_actions, log_prob, entropy, means, _, _ = self.mac.agent.sample(states,None,reparametrized=True) #If using RNN, None should be the hidden states and should be initialized


        adv = self.calc_advantage(states, means, reparam_actions)
       
        mix_loss = (log_prob * adv).mean()

        # Adaptive Entropy Regularization
        entropy_loss = entropy.sum(dim=-1).mean()
        epsilon = 1e-6
        entropy_ratio = self.entropy_coef / (entropy_loss.item() + epsilon)
        mix_loss = - mix_loss - entropy_ratio * entropy_loss

        # Optimise agents
        self.agent_optimiser.zero_grad()
        mix_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if t_env - self.log_stats_t_agent >= self.args.learner_log_interval:
            self.logger.log_stat("mix_loss", mix_loss.item(), t_env)
            self.logger.log_stat("entropy", entropy_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("entropy_ratio", entropy_ratio, t_env)
            self.logger.log_stat("entropy_loss", entropy_ratio * entropy_loss.item(), t_env)

            actor_train = '=> [{}, Episode {} t: {}] Actor:  mix_loss: {:.4f}, entropy: {:.4f}, agent_grad_norm: {:.4f}, entropy_ratio: {:.4f}, entropy_loss: {:.4f} \n'.format(datetime.now(),episode_num, t_env, mix_loss.item(), entropy_loss.item(), grad_norm, entropy_ratio, entropy_ratio * entropy_loss.item())
            self.f.write(actor_train) 
            self.f.flush() 
            self.logger.tb_logger.flush()
            self.log_stats_t_agent = t_env


    def calc_q_targets(self, batch):
        '''Q_tar = r + gamma * (target_Q(s', a'))'''
        states, actions, rewards, next_states, dones = batch  
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)
        with th.no_grad():
            next_actions, _, _, _, _, _ = self.mac.agent.sample(next_states,None)

            next_target_q1_preds = self.target_critic_q1(next_actions, next_states)
            next_target_q2_preds = self.target_critic_q2(next_actions, next_states)
              
            next_target_q_preds = th.min(next_target_q1_preds, next_target_q2_preds)
            q_targets = rewards + self.args.gamma * (1 - dones) * (next_target_q_preds)
        return q_targets
        

    def train_critic(self, batch, t_env, episode):

        states, actions, rewards, next_states, dones = batch  

        q_targets = self.calc_q_targets(batch)
        # Q-value loss for both Q nets
        q1_preds = self.critic_q1(actions, states)
        q1_error = (q1_preds - q_targets)
        q1_loss  = (q1_error ** 2).mean()

        self.critic_q1_optimiser.zero_grad()
        q1_loss.backward()
        grad_norm_q1 = th.nn.utils.clip_grad_norm_(self.critic_params_q1, self.args.grad_norm_clip)
        self.critic_q1_optimiser.step()
        
        q2_preds = self.critic_q2(actions, states)
        q2_error = (q2_preds - q_targets)
        q2_loss  = (q2_error ** 2).mean()

        self.critic_q2_optimiser.zero_grad()
        q2_loss.backward()
        grad_norm_q2 = th.nn.utils.clip_grad_norm_(self.critic_params_q2, self.args.grad_norm_clip)
        self.critic_q2_optimiser.step()

        self.critic_training_steps += 1

        if t_env - self.log_stats_t >= self.args.learner_log_interval:

            self.logger.log_stat("critic_q1_loss", q1_loss.item(), t_env)
            self.logger.log_stat("critic_q2_loss", q2_loss.item(), t_env)

            self.logger.log_stat("q1_error_abs", q1_error.abs().sum().item(), t_env)
            self.logger.log_stat("q2_error_abs", q2_error.abs().sum().item(), t_env)

            self.logger.log_stat("q1_grad_norm", grad_norm_q1, t_env)
            self.logger.log_stat("q2_grad_norm", grad_norm_q2, t_env)

            self.logger.log_stat("q1_preds_mean", q1_preds.mean().item(), t_env)
            self.logger.log_stat("q2_preds_mean", q2_preds.mean().item(), t_env)

            self.logger.log_stat("q_targets_mean", q_targets.mean().item(), t_env)

            critic_train = "=> [Episode {} t: {} ] Critic:  critic_q1_loss: {:.4f}, critic_q2_loss: {:.4f}, q1_error_abs: {:.4f}, q2_error_abs: {:.4f}, ".format(episode, t_env, q1_loss.item(), q2_loss.item(), q1_error.abs().sum().item(), q2_error.abs().sum().item())
            critic_train += "q1_grad_norm: {:.4f}, q2_grad_norm: {:.4f}, q1_preds_mean: {:.4f}, q2_preds_mean: {:.4f}, q_targets_mean: {:.4f}".format(grad_norm_q1, grad_norm_q2, q1_preds.mean().item(), q2_preds.mean().item(), q_targets.mean().item())
            critic_train += "\n"
            self.f.write(critic_train) 
            self.f.flush() 
            self.logger.tb_logger.flush()
            self.log_stats_t = t_env

        # Update target critic
        self._update_targets(polyak_update=True)
    
    def _update_targets(self,polyak_update=False, tau=0.005):
        if polyak_update:
            self.polyak_update(self.target_critic_q1, self.critic_q1, tau=tau)
            self.polyak_update(self.target_critic_q2, self.critic_q2, tau=tau)
        else:
            self.target_critic_q1.load_state_dict(self.critic_q1.state_dict())
            self.target_critic_q2.load_state_dict(self.critic_q2.state_dict())
            self.logger.console_logger.info("Updated target network")


    def polyak_update(self, target_network, source_network, tau=0.005):
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
    

    def cuda(self):
        self.mac.cuda()
        self.critic_q1.cuda()
        self.critic_q2.cuda()
        self.target_critic_q1.cuda()
        self.target_critic_q2.cuda()

    def debug_file_wandb(self, wandb, debug_output_file):
        self._wandb = wandb
        self._debug_output_file = debug_output_file
        self.f = open(self._debug_output_file, "a")

    def save_models(self, path, best=False):
        self.mac.save_models(path, best=best)
        critic_q1_str, critic_q2_str, critic_q1_opt_str, critic_q2_opt_str, agent_opt_str = self.get_strings_for_load_and_save(best=best)

        th.save(self.critic_q1.state_dict(), critic_q1_str.format(path))
        th.save(self.critic_q2.state_dict(), critic_q2_str.format(path))

        th.save(self.agent_optimiser.state_dict(), agent_opt_str.format(path))
        th.save(self.critic_q1_optimiser.state_dict(), critic_q1_opt_str.format(path))
        th.save(self.critic_q2_optimiser.state_dict(), critic_q2_opt_str.format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        critic_q1_str, critic_q2_str, critic_q1_opt_str, critic_q2_opt_str, agent_opt_str = self.get_strings_for_load_and_save(best=self.args.load_best)
        
        self.critic_q1.load_state_dict(th.load(critic_q1_str.format(path), map_location=lambda storage, loc: storage))
        self.critic_q2.load_state_dict(th.load(critic_q2_str.format(path), map_location=lambda storage, loc: storage))

        self.target_critic_q1.load_state_dict(self.critic_q1.state_dict())
        self.target_critic_q2.load_state_dict(self.critic_q2.state_dict())

        self.agent_optimiser.load_state_dict(th.load(agent_opt_str.format(path), map_location=lambda storage, loc: storage))
        self.critic_q1_optimiser.load_state_dict(th.load(critic_q1_opt_str.format(path), map_location=lambda storage, loc: storage))
        self.critic_q2_optimiser.load_state_dict(th.load(critic_q2_opt_str.format(path), map_location=lambda storage, loc: storage))

    def get_strings_for_load_and_save(self,best=False):
        prefix_str = "{}/"
        critic_q1_str = "critic_q1"
        critic_q2_str = "critic_q2"
        extension_str = ".th"
        opt_str = "opt"
        agent_opt_str = prefix_str + "agent" + "_" + opt_str
        critic_q1_opt_str = prefix_str + critic_q1_str + "_" + opt_str
        critic_q2_opt_str = prefix_str + critic_q2_str + "_" + opt_str
        critic_q1_str = prefix_str + critic_q1_str
        critic_q2_str = prefix_str + critic_q2_str

        if best:
            agent_opt_str += "_best" 
            critic_q1_opt_str += "_best"
            critic_q2_opt_str += "_best"
            critic_q1_str += "_best"
            critic_q2_str += "_best"

        agent_opt_str +=  extension_str
        critic_q1_opt_str += extension_str
        critic_q2_opt_str += extension_str
        critic_q1_str += extension_str
        critic_q2_str += extension_str

        return critic_q1_str, critic_q2_str, critic_q1_opt_str, critic_q2_opt_str, agent_opt_str
    
    def init_order(self):
        self.order = [_ for _ in range(self.n_agents)]

    