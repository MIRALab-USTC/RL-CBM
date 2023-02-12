

from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn
import torch.functional as F
from torch.distributions import Normal

import cbm.torch_modules.utils as ptu
from cbm.utils.misc_untils import get_scheduled_value
from cbm.agents.ddpg_agent import DDPGAgent
from cbm.agents.drq.drq_agent import log_frame
from cbm.processors.utils import RandomShiftsAug
from cbm.utils.logger import logger

import numpy as np
import pdb
import copy

class DrQv2Agent(DDPGAgent): 
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        processor,
        n_step_td=3,
        n_aug=2, 
        image_pad=4,
        image_aug=True,
        noise_schedule=[0,100000,1,0.1],
        debug_loss_threshold=None,
        **sac_agent_kwargs
    ):
        #TRY: with weight
        self.action_repeat = env.action_repeat
        self.frame_stack = env.frame_stack
        self.n_step_td = n_step_td
        self.n_aug = n_aug
        self.aug_trans = RandomShiftsAug(image_pad)
        self.processor = processor
        sample_kwargs = {
            'deterministic': False,
            'use_noise_clip': True,
        }
        DDPGAgent.__init__(
            self,
            env,
            policy,
            qf,
            qf_target,
            next_sample_kwargs=sample_kwargs,
            current_sample_kwargs=sample_kwargs,
            **sac_agent_kwargs
        )
        self._true_discount = self.discount
        self.discount = self.discount ** n_step_td
        self.noise_schedule = noise_schedule
        self.critic_params = list(self.qf.parameters()) + list(self.processor.parameters())
        self.qf_optimizer = self.optimizer_class(
            self.critic_params,
            lr=self.qf_lr,
            **self.optimizer_kwargs
        )
        if debug_loss_threshold is None:
            self.debug_loss_threshold = debug_loss_threshold 
        else:
            self.debug_loss_threshold = 500
        if logger.log_or_not(logger.DEBUG):
            self.anomaly_count = 0

    @property
    def num_init_steps(self):
        return self._num_init_steps * self.action_repeat

    @property
    def num_explore_steps(self):
        return self._num_explore_steps * self.action_repeat

    def set_noise_scale(self):
        noise_scale = get_scheduled_value(
            self.num_total_steps, 
            self.noise_schedule
        )
        self.policy.set_noise_scale(noise_scale)

    def step_explore(self, o, **kwargs):
        o = ptu.from_numpy(o)
        o = self.processor(o)
        with self.policy.deterministic_(False):
            a, _ = self.policy.action(o, use_noise_clip=False, **kwargs)
        a = ptu.get_numpy(a)
        return a, {}

    def step_exploit(self, o, **kwargs):
        o = ptu.from_numpy(o)
        o = self.processor(o)
        with self.policy.deterministic_(True):
            a, _ = self.policy.action(o, **kwargs)
        a = ptu.get_numpy(a)
        return a, {}     
    
    def compute_q_target(self, actor_next_obs, next_obs, rewards, terminals, v_pi_kwargs={}):
        with torch.no_grad():
            next_action, _ = self.policy.action(actor_next_obs, **self.next_sample_kwargs)
            target_q_next_action = self.qf_target.value(
                next_obs, next_action, **v_pi_kwargs
            )[0]
            q_target = rewards + (1.-terminals)*self.discount*target_q_next_action
        return q_target

    def log_frame(self, frame, next_frame):
        if self._log_tb_or_not() and logger.log_or_not(logger.INFO):
            max_diff, index = torch.max(self.debug_info["diff"], dim=1)
            for i, (md, ind) in enumerate(zip(max_diff, index)):
                logger.tb_add_scalar('diff/q%d_max'%i, md, self.num_train_steps)
                # log_frame(frame, ind, 'q%d_maxdiff/cur_obs'%i, self.num_train_steps)
                # log_frame(next_frame, ind, 'q%d_maxdiff/next_obs'%i, self.num_train_steps)
            min_diff, index = torch.min(self.debug_info["diff"], dim=1)
            for i, (md, ind) in enumerate(zip(min_diff, index)):
                logger.tb_add_scalar('diff/q%d_min'%i, md, self.num_train_steps)
                # log_frame(frame, ind, 'q%d_mindiff/cur_obs'%i, self.num_train_steps)
                # log_frame(next_frame, ind, 'q%d_mindiff/next_obs'%i, self.num_train_steps)
            logger.tb_flush()
            
    def train_from_torch_batch(self, batch):
        self.set_noise_scale()
        self.train_info['noise_scale'] = self.policy.noise_scale
        ######################################
        # obtain n-step data to train models #
        ######################################
        #check
        shift = self.n_step_td
        frame_stack = self.frame_stack
        frames = batch['frames']
        assert frames.shape[1] == frame_stack + shift
        C, H, W = frames.shape[2], frames.shape[3], frames.shape[4]
        cur_frames = frames[:,:frame_stack].reshape(-1, C*frame_stack, H, W)
        next_frames = frames[:,-frame_stack:].reshape(-1, C*frame_stack, H, W)
        actions = batch['actions'][:,frame_stack-1]
        discount = 1
        live = 1
        rewards = 0
        for i in range(shift):
            rewards = rewards + batch['rewards'][:,i]*discount
            live = live * (1-batch['terminals'][:,i])
            discount = discount * self._true_discount * live
        terminals = 1-live
        self.log_batch(rewards, terminals)
        ################
        # augmentation #
        ################
        batch_size = len(next_frames)
        frame_aug = []
        next_frame_aug = []
        rewards_aug = []
        terminals_aug = []
        actions_aug = []
        for _ in range(self.n_aug):
            frame_aug.append(self.aug_trans(cur_frames))
            next_frame_aug.append(self.aug_trans(next_frames))
            rewards_aug.append(rewards)
            terminals_aug.append(terminals)
            actions_aug.append(actions)
        frame_aug = torch.cat(frame_aug, dim=0)
        next_frame_aug = torch.cat(next_frame_aug, dim=0)
        rewards_aug = torch.cat(rewards_aug, dim=0)
        terminals_aug = torch.cat(terminals_aug, dim=0)
        actions_aug = torch.cat(actions_aug, dim=0)
        # process frames
        obs_aug = self.processor(frame_aug)
        next_obs_aug = self.processor(next_frame_aug)
        actor_next_obs_aug = next_obs_aug
        #################
        # update critic #
        #################
        # compute target
        q_target = self.compute_q_target(
            actor_next_obs_aug,
            next_obs_aug, 
            rewards_aug, 
            terminals_aug, 
            self.next_v_pi_kwargs
        )
        q_target = q_target.reshape((self.n_aug,batch_size,1)).mean(0)
        q_target_aug = []
        for _ in range(self.n_aug):
            q_target_aug.append(q_target)
        q_target_aug = torch.cat(q_target_aug, dim=0)
        # compute loss
        qf_loss, train_qf_info = self.compute_qf_loss(
            obs_aug, 
            actions_aug, 
            q_target_aug
        )
        self.log_frame(frame_aug, next_frame_aug)
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
            actor_obs = obs_aug[:batch_size].detach()
            new_action, action_info = self.policy.action(
                actor_obs, **self.current_sample_kwargs
            )
            policy_loss, train_policy_info = self.compute_policy_loss(
                actor_obs,  #note
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
        if logger.log_or_not(logger.DEBUG):
            self.anomaly_detection()
        batch_keys = list(batch.keys())
        for k in batch_keys:
            del batch[k]
        self.log_train_info()
        self.num_train_steps += 1
        return copy.deepcopy(self.train_info)


    @property
    def networks(self):
        nets = [
            self.policy,
            self.qf,
            self.processor
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf=self.qf,
            obs_processor=self.processor
        )
    
    def anomaly_detection(self):
        if self.debug_info['qf_loss'].item() > self.debug_loss_threshold:
            return True
        else:
            return False
        

