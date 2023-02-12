from collections import OrderedDict

import numpy as np
from numpy.core.fromnumeric import argmin
import torch
from cbm.agents.drq.drq_agent import DrQAgent, log_frame
from cbm.processors.utils import *
from cbm.utils.logger import logger
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import copy


class ProtoAgent(DrQAgent): 
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        processor,
        proto_model,
        model_coef=1,
        policy_share_trunk=True,
        qf_share_trunk=True,
        detach_qf_obs=True,
        noise_schedule=None,
        **drq_kwargs
    ):
        super().__init__(env, policy, qf, qf_target,processor,**drq_kwargs)
        self.proto_model = proto_model
        if policy_share_trunk:
            policy.trunk = proto_model.trunk
            self.policy_optimizer = self.optimizer_class(
                self.policy.module.parameters(),
                lr=self.policy_lr,
                **self.optimizer_kwargs
            )
        else: 
            assert policy.trunk_detach == False
        if qf_share_trunk:
            qf.trunk = proto_model.trunk
        else:
            assert qf.trunk_detach == False 
        self.model_coef = model_coef
        combination_parameters = []
        combination_parameters += qf.parameters()
        combination_parameters += processor.parameters()
        combination_parameters += proto_model.parameters()
        self.combination_optimizer = self.optimizer_class(
            combination_parameters,
            lr=self.qf_lr,
            **self.optimizer_kwargs
        )
        self._true_discount = self.discount
        self.combination_lr = self.qf_lr
        self.detach_qf_obs = detach_qf_obs
        self.action_repeat = env.action_repeat
        self.r_seq = proto_model.r_seq

    def train_from_torch_batch(self, batch):
        frame_stack = self.frame_stack
        frames = batch['frames']
        C, H, W = frames.shape[2], frames.shape[3], frames.shape[4]
        cur_frames = frames[:,:frame_stack].reshape(-1, C*frame_stack, H, W)
        next_frames = frames[:,1:1+frame_stack].reshape(-1, C*frame_stack, H, W)
        rewards = batch['rewards'][:,frame_stack-1]
        terminals = batch['terminals'][:,frame_stack-1]
        actions = batch['actions'][:,frame_stack-1]
        self.log_batch(rewards, terminals)
        # for auxiliary model
        discount = 1
        live = 1
        aux_r = 0

        for i in range(self.r_seq):
            aux_r = aux_r + batch['rewards'][:,i]*discount
            live = live * (1-batch['terminals'][:,i])
            discount = discount * self._true_discount * live

        ################
        # augmentation #
        ################
        batch_size = len(next_frames)
        frame_aug = []
        next_frame_aug = []
        for _ in range(self.n_aug):
            sep_aug = False
            if sep_aug:
                frame_aug.append(self.sep_aug(cur_frames))
                next_frame_aug.append(self.sep_aug(next_frames))
            else:
                frame_aug.append(self.aug_trans(cur_frames))
                next_frame_aug.append(self.aug_trans(next_frames))
        frame_aug = torch.cat(frame_aug, dim=0)
        next_frame_aug = torch.cat(next_frame_aug, dim=0)
        rewards_aug = rewards.repeat(self.n_aug,1)
        terminals_aug = terminals.repeat(self.n_aug,1)
        actions_aug = actions.repeat(self.n_aug,1)
        aux_r_aug = aux_r.repeat(self.n_aug,1)
        #process frames
        cat_frame_aug = torch.cat([frame_aug,next_frame_aug])
        cat_obs_aug = self.processor(cat_frame_aug)
        obs_aug, actor_next_obs_aug = torch.chunk(cat_obs_aug,2)
        if self.target_processor is None:
            next_obs_aug = actor_next_obs_aug
        else:
            next_obs_aug = self.target_processor(next_frame_aug)
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
        q_target_aug = q_target.repeat(self.n_aug,1)
        # compute loss
        if self.detach_qf_obs:
            value_obs_aug = obs_aug.detach()
        else:
            value_obs_aug = obs_aug
        qf_loss, train_qf_info = self.compute_qf_loss(
            value_obs_aug, 
            actions_aug, 
            q_target_aug)
        self.train_info.update(train_qf_info)
        self.log_frame(frame_aug, next_frame_aug)
        ################
        # update model #
        ################
        model_loss = self.proto_model.compute_auxiliary_loss(
            obs_aug, actions_aug, aux_r_aug, next_obs_aug, q_target_aug, 
            n_step=self.num_train_steps, 
            log=self._log_tb_or_not(), 
            next_frame=next_frame_aug)
        ############## end ##############
        self.combination_optimizer.zero_grad()
        (qf_loss+self.model_coef*model_loss).backward()
        self.log_critic_grad_norm()
        self.combination_optimizer.step()
        if logger.log_or_not(logger.INFO) and self._log_tb_or_not():
            self.plot_value_scatter()
        if self.num_train_steps % self.target_update_freq == 0:
            self._update_target(self.soft_target_tau)
        ###########################
        # update actor and alpha #
        ###########################
        if self.num_train_steps % self.policy_update_freq == 0:
            actor_obs = obs_aug.detach()
            new_action, action_info = self.policy.action(
                actor_obs, **self.current_sample_kwargs
            )
            # update alpha
            if self.use_automatic_entropy_tuning:
                alpha_loss, train_alpha_info = self.compute_alpha_loss(action_info)
                self.train_info.update(train_alpha_info)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
            # update policy
            policy_loss, train_policy_info = self.compute_policy_loss(
                actor_obs,  #note
                new_action, 
                action_info, 
                self.current_v_pi_kwargs
            )
            self.train_info.update(train_policy_info)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
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