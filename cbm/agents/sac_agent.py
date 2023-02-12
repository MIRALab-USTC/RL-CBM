from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim 

import cbm.torch_modules.utils as ptu
from cbm.agents.ddpg_agent import DDPGAgent
from cbm.utils.eval_util import create_stats_ordered_dict
from cbm.utils.logger import logger
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import copy

class SACAgent(DDPGAgent): 
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        policy_lr=3e-4,
        alpha_if_not_automatic=1e-2,
        use_automatic_entropy_tuning=True,
        init_log_alpha=0,
        target_entropy=None,
        next_sample_kwargs={
            'reparameterize':False, 
            'return_log_prob':True
        },
        current_sample_kwargs={
            'reparameterize':True, 
            'return_log_prob':True
        },
        **ddpg_kwargs
    ):
        if logger.log_or_not(logger.INFO):
            current_sample_kwargs['return_mean_std'] = True
        super().__init__(
            env,
            policy,
            qf,
            qf_target,
            policy_lr=policy_lr,
            next_sample_kwargs=next_sample_kwargs,
            current_sample_kwargs=current_sample_kwargs,
            **ddpg_kwargs
        )
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.alpha_if_not_automatic = alpha_if_not_automatic
        if self.use_automatic_entropy_tuning:
            if target_entropy is not None:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  
            self.log_alpha = ptu.FloatTensor([init_log_alpha])
            self.log_alpha.requires_grad_(True)
            self.alpha_optimizer = self.optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
                **self.optimizer_kwargs
            )

    def _get_alpha(self):
        if self.use_automatic_entropy_tuning:
            with torch.no_grad():
                alpha = self.log_alpha.exp() 
        else:
            alpha = self.alpha_if_not_automatic
        return alpha

    def compute_q_target(self, next_obs, rewards, terminals, v_pi_kwargs={}):
        with torch.no_grad():
            alpha = self._get_alpha()
            next_action, next_policy_info = self.policy.action(
                next_obs, **self.next_sample_kwargs
            )
            log_prob_next_action = next_policy_info['log_prob']
            target_q_next_action = self.qf_target.value(
                next_obs, 
                next_action, 
                **v_pi_kwargs
            )[0] - alpha * log_prob_next_action
            q_target = rewards + (1.-terminals)*self.discount*target_q_next_action
        return q_target

    def compute_alpha_loss(self, action_info, prefix='alpha/'):
        assert self.use_automatic_entropy_tuning
        log_prob_new_action = action_info['log_prob']
        res = (log_prob_new_action.mean() + self.target_entropy).detach()
        # if res>0, means terget_entropy > current_entropy, expect a larger alpha
        alpha = self.log_alpha.exp()
        alpha_loss = (-res) * alpha
        # compute sttistics
        alpha_info = OrderedDict()
        alpha_info[prefix+'value'] = alpha.item()
        alpha_info[prefix+'loss'] = alpha_loss.item()
        return alpha_loss, alpha_info

    def compute_policy_loss(self, obs, new_action, action_info, v_pi_kwargs={}, prefix='policy/'):
        log_prob_new_action = action_info['log_prob']
        alpha = self._get_alpha()
        q_new_action, _ = self.qf.value(
            obs, 
            new_action, 
            return_ensemble=False, 
            **v_pi_kwargs
        )
        entropy = -log_prob_new_action.mean()
        q_pi_mean = q_new_action.mean()
        policy_loss = -alpha*entropy - q_pi_mean
        policy_info = self._log_policy_info(
            new_action, action_info, policy_loss,
            q_pi_mean, entropy, prefix)
        return policy_loss, policy_info
    
    def _log_policy_info(self, 
        new_action, action_info, policy_loss, 
        q_pi_mean, entropy, prefix):
        policy_info = OrderedDict()
        log_prob_new_action = action_info['log_prob']
        if logger.log_or_not(logger.INFO):
            policy_info.update(create_stats_ordered_dict(
                prefix+'log_prob',
                ptu.get_numpy(log_prob_new_action),
            ))
        if self._log_tb_or_not(): 
            if logger.log_or_not(logger.INFO):
                all_mean = action_info['mean'].detach().cpu().numpy()
                all_std = action_info['std'].detach().cpu().numpy()
            for i in range(new_action.shape[-1]):
                logger.tb_add_histogram(
                    'action_%d'%i, 
                    new_action[:,i], 
                    self.num_train_steps
                )
                if logger.log_or_not(logger.INFO):
                    mean = all_mean[...,i]
                    std = all_std[...,i]
                    fig, ax = plt.subplots()
                    ax.grid(ls='--')
                    ax.scatter(mean, std, alpha=0.2)
                    logger.tb_add_figure("action/dim_%d/mean_vs_std"%i, fig, self.num_train_steps)
            logger.tb_add_histogram('log_prob_new_action', log_prob_new_action, self.num_train_steps)
            logger.tb_flush()
        policy_info[prefix+'loss'] = policy_loss.item()
        policy_info[prefix+'q_pi'] = q_pi_mean.item()
        policy_info[prefix+'entropy'] = entropy.item()
        return policy_info

    def train_from_torch_batch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        self.log_batch(rewards, terminals)
        #################
        # update critic #
        #################
        q_target = self.compute_q_target(next_obs, rewards, terminals, self.next_v_pi_kwargs)
        qf_loss, train_qf_info = self.compute_qf_loss(obs, actions, q_target)
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.log_critic_grad_norm()
        self.qf_optimizer.step()
        if logger.log_or_not(logger.INFO) and self._log_tb_or_not():
            self.plot_value_scatter()
        if self.num_train_steps % self.target_update_freq == 0:
            self._update_target(self.soft_target_tau)
        self.train_info.update(train_qf_info)
        ###########################
        # update actor and alpha #
        ###########################
        if self.num_train_steps % self.policy_update_freq == 0:
            new_action, action_info = self.policy.action(
                obs, **self.current_sample_kwargs
            )
            #update alpha
            if self.use_automatic_entropy_tuning:
                alpha_loss, train_alpha_info = self.compute_alpha_loss(action_info)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.train_info.update(train_alpha_info)
            #update policy
            policy_loss, train_policy_info = self.compute_policy_loss(
                obs, 
                new_action, 
                action_info, 
                self.current_v_pi_kwargs
            )
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            self.train_info.update(train_policy_info)
        #####################
        # update statistics #
        #####################
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics.update(self.train_info)
        self.log_train_info()
        self.num_train_steps += 1
        return copy.deepcopy(self.train_info)
