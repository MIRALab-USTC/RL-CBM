import torch
from cbm.agents.drqv2.drqv2_agent import DrQv2Agent
from cbm.processors.utils import *
from cbm.utils.logger import logger
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import copy

class Protov2Agent(DrQv2Agent): 
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
        **drq_kwargs
    ):
        super().__init__(env, policy, qf, qf_target,processor,**drq_kwargs)
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
        self.proto_model = proto_model
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
        self.combination_lr = self.qf_lr
        self.detach_qf_obs = detach_qf_obs
        self.action_repeat = env.action_repeat
        self.r_seq = proto_model.r_seq

    def train_from_torch_batch(self, batch):
        self.set_noise_scale()
        self.train_info['noise_scale'] = self.policy.noise_scale
        ######################################
        # obtain n-step data to train models #
        ######################################
        shift = self.n_step_td
        frame_stack = self.frame_stack
        frames = batch['frames']
        assert frames.shape[1] == frame_stack + shift
        C, H, W = frames.shape[2], frames.shape[3], frames.shape[4]
        cur_frames = frames[:,:frame_stack].reshape(-1, C*frame_stack, H, W)
        next_frames = frames[:,-frame_stack:].reshape(-1, C*frame_stack, H, W)
        actions = batch['actions'][:,frame_stack-1]
        aux_next_frames = frames[:,1:1+frame_stack].reshape(-1,3*C,H,W)
        aux_next_actions = batch['actions'][:,frame_stack]
        discount = 1
        live = 1
        aux_r = 0
        # for auxiliary model
        
        for i in range(self.r_seq):
            aux_r = aux_r + batch['rewards'][:,i]*discount
            live = live * (1-batch['terminals'][:,i])
            discount = discount * self._true_discount * live
        
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
        aux_next_frame_aug = []
        for _ in range(self.n_aug):
            frame_aug.append(self.aug_trans(cur_frames))
            next_frame_aug.append(self.aug_trans(next_frames))
            aux_next_frame_aug.append(self.aug_trans(aux_next_frames))
        frame_aug = torch.cat(frame_aug, dim=0)
        next_frame_aug = torch.cat(next_frame_aug, dim=0)
        aux_next_frame_aug = torch.cat(aux_next_frame_aug, dim=0)
        rewards_aug = rewards.repeat(self.n_aug,1)
        terminals_aug = terminals.repeat(self.n_aug,1)
        actions_aug = actions.repeat(self.n_aug,1)
        aux_next_actions_aug = aux_next_actions.repeat(self.n_aug,1)
        aux_r_aug = aux_r.repeat(self.n_aug,1)
        # process frames
        cat_frame_aug = torch.cat([frame_aug, next_frame_aug, aux_next_frame_aug])
        cat_obs_aug = self.processor(cat_frame_aug)
        obs_aug, actor_next_obs_aug, aux_next_obs_aug = torch.chunk(cat_obs_aug,3)
        next_obs_aug = actor_next_obs_aug
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
            obs_aug, actions_aug, aux_r_aug, aux_next_obs_aug, aux_next_actions_aug, 
            n_step=self.num_train_steps, 
            log=self._log_tb_or_not(), 
            next_frame=next_frame_aug)
        ############## end ##############
        # self.qf_optimizer.zero_grad()
        # qf_loss.backward()
        # self.log_critic_grad_norm()
        # self.qf_optimizer.step()
        self.combination_optimizer.zero_grad()
        (qf_loss+self.model_coef*model_loss).backward()
        self.log_critic_grad_norm()
        self.combination_optimizer.step()
        if logger.log_or_not(logger.INFO) and self._log_tb_or_not():
            self.plot_value_scatter()
        if self.num_train_steps % self.target_update_freq == 0:
            self._update_target(self.soft_target_tau)
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
            self.processor,
            self.proto_model
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf=self.qf,
            obs_processor=self.processor,
            aux_net=self.proto_model
        )