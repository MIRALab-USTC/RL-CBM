from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim 
import torch.nn.functional as F
from torch import nn

import cbm.torch_modules.utils as ptu
from cbm.utils.eval_util import create_stats_ordered_dict
from cbm.agents.base_agent import BatchTorchAgent
from cbm.utils.logger import logger
import copy
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def plot_scatter(x, y, extra_line, name, n_step):
    x = x.detach().cpu().numpy()
    min_x, max_x = x.min(), x.max()
    y = y.detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.grid(ls='--')
    if len(y.shape) == len(x.shape) + 1:
        for y_ in y:
            ax.scatter(x, y_, alpha=0.2)
    elif len(y.shape) == len(x.shape):
        ax.scatter(x, y, s=5, alpha=0.2)
    else:
        raise NotImplementedError
    if extra_line == 'no':
        pass
    elif extra_line == 'x=y':
        ax.plot([min_x,max_x],[min_x,max_x], color='gray', ls='--', alpha=0.5)
    elif extra_line == 'zero':
        ax.plot([min_x,max_x],[0, 0], color='gray', ls='--', alpha=0.5)
    else:
        raise NotImplementedError
    logger.tb_add_figure(name, fig, n_step)

class DDPGAgent(BatchTorchAgent): 
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        num_seed_steps=-1,
        discount=0.99,
        policy_lr=3e-4,
        qf_lr=3e-4,
        soft_target_tau=5e-3,
        policy_update_freq=1,
        target_update_freq=1,
        clip_error=None,
        optimizer_class='Adam',
        optimizer_kwargs={},
        next_sample_kwargs={},
        current_sample_kwargs={}, 
        next_v_pi_kwargs={}, 
        current_v_pi_kwargs={}, 
        tb_log_freq=100,
    ):
        super().__init__(env)
        if isinstance(optimizer_class, str):
            self.optimizer_class = eval('optim.'+optimizer_class)
        self.optimizer_kwargs = optimizer_kwargs
        self.policy = policy
        self.qf = qf
        self.qf_target =qf_target
        self._update_target(1)
        self.num_train_steps = 0
        self.num_seed_steps = num_seed_steps
        self.soft_target_tau = soft_target_tau
        self.target_update_freq = target_update_freq
        self.policy_update_freq = policy_update_freq
        self.clip_error = clip_error

        self.policy_optimizer = self.optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
            **self.optimizer_kwargs
        )
        self.qf_optimizer = self.optimizer_class(
            self.qf.parameters(),
            lr=qf_lr,
            **self.optimizer_kwargs
        )
        self.critic_params = list(self.qf.parameters())
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr

        self.discount = discount
        self.train_info = OrderedDict()
        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True

        self.next_sample_kwargs = next_sample_kwargs
        self.current_sample_kwargs = current_sample_kwargs
        self.next_v_pi_kwargs = next_v_pi_kwargs
        self.current_v_pi_kwargs = current_v_pi_kwargs
        self.tb_log_freq = tb_log_freq

        if logger.log_or_not(logger.INFO):
            self.debug_info = {}

    def step_init(self, o, **kwargs):
        if self.num_seed_steps<0 or self.num_total_steps < self.num_seed_steps:
            return super().step_init(o, **kwargs)
        else:
            return self.step_explore(o, **kwargs)

    def _log_tb_or_not(self):
        if self.tb_log_freq > 0 and logger.log_or_not(logger.ERROR) \
            and (self.num_train_steps % self.tb_log_freq==0):
            return True
        else:
            return False

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def _update_target(self, tau):
        if tau == 1:
            ptu.copy_model_params_from_to(self.qf, self.qf_target)
        else:
            ptu.soft_update_from_to(self.qf, self.qf_target, tau)

    def compute_q_target(self, next_obs, rewards, terminals, v_pi_kwargs={}):
        with torch.no_grad():
            next_action, _ = self.policy.action(next_obs, **self.next_sample_kwargs)
            target_q_next_action = self.qf_target.value(
                next_obs, 
                next_action, 
                **v_pi_kwargs
            )[0]
            q_target = rewards + (1.-terminals)*self.discount*target_q_next_action
        return q_target

    def compute_qf_loss(self, obs, actions, q_target, prefix='qf/'):
        _, value_info = self.qf.value(obs, actions, return_ensemble=True)
        q_value_ensemble = value_info['ensemble_value']
        q_target_expand = q_target.detach().expand(q_value_ensemble.shape)
        if self.clip_error is not None and self.clip_error > 0:
            with torch.no_grad():
                diff = q_target_expand-q_value_ensemble
                diff = torch.clamp(diff, -self.clip_error, self.clip_error)
                q_target_expand = (diff + q_value_ensemble).detach()
        qf_loss = F.mse_loss(q_value_ensemble, q_target_expand)
        qf_info = self._log_q_info(q_target, q_value_ensemble, qf_loss, prefix)
        # assert torch.isnan(qf_loss)==False
        return qf_loss, qf_info

    def _log_q_info(self, q_target, q_value_ensemble, qf_loss, prefix, tb_prefix=""):
        """
        LOG FOR Q FUNCTION
        """
        qf_info = OrderedDict()
        qf_info[prefix+'loss'] = np.mean(ptu.get_numpy(qf_loss))
        if logger.log_or_not(logger.WARNING):
            qf_info.update(create_stats_ordered_dict(
                    prefix+'target',
                    ptu.get_numpy(q_target),
                ))
            q_pred_mean = torch.mean(q_value_ensemble, dim=0)
            qf_info.update(create_stats_ordered_dict(
                prefix+'pred_mean',
                ptu.get_numpy(q_pred_mean),
            ))
            q_pred_std = torch.std(q_value_ensemble, dim=0)
            qf_info.update(create_stats_ordered_dict(
                prefix+'pred_std',
                ptu.get_numpy(q_pred_std),
            ))

        if self._log_tb_or_not() or logger.log_or_not(logger.INFO):
            diff = q_value_ensemble - q_target
        if logger.log_or_not(logger.INFO):
            self.debug_info['q_value_ensemble'] = q_value_ensemble
            self.debug_info['q_target'] = q_target
            self.debug_info['diff'] = diff
            self.debug_info['qf_loss'] = qf_loss
        if self._log_tb_or_not():
            logger.tb_add_histogram('diff', diff[0], self.num_train_steps)
            logger.tb_add_histogram('q_value_ensemble', q_value_ensemble[0], self.num_train_steps)
            logger.tb_add_histogram('q_target', q_target, self.num_train_steps)
            logger.tb_flush()
        return qf_info

    def compute_policy_loss(self, obs, new_action, action_info, v_pi_kwargs={}, prefix='policy/'):
        q_new_action, _ = self.qf.value(
            obs, 
            new_action, 
            return_ensemble=False, 
            **v_pi_kwargs
        )
        q_pi_mean = q_new_action.mean()
        policy_loss = -q_pi_mean
        policy_info = self._log_policy_info(
            new_action, policy_loss, q_pi_mean, prefix)
        # assert torch.isnan(policy_loss)==False
        return policy_loss, policy_info

    def _log_policy_info(self, new_action, policy_loss, q_pi_mean, prefix):
        policy_info = OrderedDict()
        policy_info[prefix+'loss'] = policy_loss.item()
        policy_info[prefix+'q_pi'] = q_pi_mean.item()
        if self._log_tb_or_not(): 
            for i in range(new_action.shape[-1]):
                logger.tb_add_histogram(
                    'action_%d'%i, 
                    new_action[:,i], 
                    self.num_train_steps
                )
            logger.tb_flush()
        return policy_info

    def log_critic_grad_norm(self):
        if self._log_tb_or_not() and logger.log_or_not(logger.WARNING):
            norm = 0
            for p in self.critic_params:
                if p.grad is not None:
                    norm = norm + p.grad.norm()
            logger.tb_add_scalar('norm/critic_grad', norm, self.num_train_steps)
        if self._log_tb_or_not() and logger.log_or_not(logger.INFO):
            self.debug_info['critic_grad_norm'] = norm.item()

    def log_batch(self, rewards, terminals):
        if self._log_tb_or_not():
            logger.tb_add_histogram('rewards', rewards, self.num_train_steps)
            logger.tb_add_histogram('terminals', terminals, self.num_train_steps)
            logger.tb_flush()

    def log_train_info(self):
        if self._log_tb_or_not() and logger.log_or_not(logger.WARNING):
            for k in self.train_info:
                logger.tb_add_scalar(k, self.train_info[k], self.num_train_steps)

    def train_from_torch_batch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        self.log_batch(rewards, terminals)
        ################
        # update critic #
        ################
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
        ################
        # update actor #
        ################
        if self.num_train_steps % self.policy_update_freq == 0:
            new_action, action_info = self.policy.action(obs, **self.current_sample_kwargs)
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

    def get_diagnostics(self):
        self.eval_statistics.update({
            "num_total_steps": self.num_total_steps,
            "num_train_steps": self.num_train_steps
        })
        return self.eval_statistics

    def plot_value_scatter(self, prefix=''):
        q_target = self.debug_info['q_target']
        diff = self.debug_info['diff']
        q_value_ensemble = self.debug_info['q_value_ensemble']
        self._plot_scatter(q_target, q_value_ensemble, 'x=y', prefix+"q_target_vs_pred_q")
        self._plot_scatter(q_target, diff, 'no', prefix+"q_target_vs_diff")
        logger.tb_flush()
    
    def _plot_scatter(self, x, y, extra_line='no', name='scatter'):
        plot_scatter(x, y, extra_line, name, self.num_train_steps)

    # NOTE these
    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.qf_target,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf=self.qf,
            qf_target=self.qf_target,
        )

