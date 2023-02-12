

from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

import cbm.torch_modules.utils as ptu
from cbm.utils.eval_util import create_stats_ordered_dict
from cbm.agents.sac_agent import SACAgent
from cbm.processors.utils import RandomShiftsAug
from cbm.utils.logger import logger
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import pdb
import copy


def log_frame(frames, index, tag, step, prefix='frames/'):
    frames = frames[index]
    H, W = frames.shape[-2:]
    frames = frames.view(-1,3,H,W)
    if (frames>1).any():
        frames = frames / 255.0
    if (frames<0).any():
        frames = frames + 0.5
    logger.tb_add_images(prefix+tag, frames, step)

    
class DrQAgent(SACAgent): 
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        processor,
        target_processor=None,
        n_aug=2, 
        image_pad=4,
        debug_loss_threshold=None,
        analyze_embeddings=False,
        projection_lr=1e-4,
        aug = True,
        detach_step = 0,
        **sac_agent_kwargs
    ):
        #TRY: with weight
        self.detach_step = detach_step
        self.action_repeat = env.action_repeat
        self.n_aug = n_aug
        self.aug_trans = RandomShiftsAug(image_pad,aug)
        self.processor = processor
        self.target_processor = target_processor
        self.frame_stack = env.frame_stack
        super().__init__(
            env,
            policy,
            qf,
            qf_target,
            **sac_agent_kwargs
        )
        self.critic_params = list(self.qf.parameters()) + list(self.processor.parameters())
        self.qf_optimizer = self.optimizer_class(
            self.critic_params,
            lr=self.qf_lr,
            **self.optimizer_kwargs
        )
        if debug_loss_threshold is not None:
            self.debug_loss_threshold = debug_loss_threshold 
        else:
            self.debug_loss_threshold = 500
        if logger.log_or_not(logger.DEBUG):
            self.anomaly_count = 0
        self.analyze_embeddings = analyze_embeddings
        if analyze_embeddings:
            assert env.return_state
            self._s_size = env.state_size
            # for latent input
            _e_size = processor.output_shape[0]
            self.linear_projection = nn.Linear(_e_size, self._s_size)
            self.linear_projection.to(ptu.device)
            self.projection_optimizer = self.optimizer_class(
                self.linear_projection.parameters(),
                lr=projection_lr,
                **self.optimizer_kwargs
            )
            self._state_dim_name = []
            for k,v in env.state_dim_dict.items():
                for i in range(v):
                    dim_name = k + ("_%d"%i)
                    self._state_dim_name.append(dim_name)
            self._s_mean, self._s_std = None, None
            self._s_fac = 0.995

    @property
    def num_init_steps(self):
        return self._num_init_steps * self.action_repeat

    @property
    def num_explore_steps(self):
        return self._num_explore_steps * self.action_repeat
    
    def step_explore(self, o, **kwargs):
        o = ptu.from_numpy(o)
        o = self.processor(o)
        a, _ = self.policy.action(o, **kwargs)
        a = ptu.get_numpy(a)
        return a, {}

    def step_exploit(self, o, **kwargs):
        o = ptu.from_numpy(o)
        o = self.processor(o)
        if hasattr(self.policy, 'deterministic_'):
            with self.policy.deterministic_(True):
                a, _ = self.policy.action(o, **kwargs)
        else:
            a, _ = self.policy.action(o, **kwargs)
        a = ptu.get_numpy(a)
        return a, {}     

    def _update_target(self, tau):
        if tau == 1:
            ptu.copy_model_params_from_to(self.qf, self.qf_target) 
            if self.target_processor is not None:
                ptu.copy_model_params_from_to(self.processor, self.target_processor) 
        else:
            ptu.soft_update_from_to(self.qf, self.qf_target, tau) 
            if self.target_processor is not None:
                ptu.soft_update_from_to(self.processor, self.target_processor, tau) 
    
    def compute_q_target(self, actor_next_obs, next_obs, rewards, terminals, v_pi_kwargs={}):
        with torch.no_grad():
            alpha = self._get_alpha()
            next_action, next_policy_info = self.policy.action(
                actor_next_obs, **self.next_sample_kwargs
            )
            log_prob_next_action = next_policy_info['log_prob']
            target_q_next_action = self.qf_target.value(
                next_obs, 
                next_action, 
                **v_pi_kwargs
            )[0] - alpha * log_prob_next_action
            q_target = rewards + (1.-terminals)*self.discount*target_q_next_action
        return q_target

    def log_frame(self, frame, next_frame):
        if self._log_tb_or_not() and logger.log_or_not(logger.INFO):
            max_diff, index = torch.max(self.debug_info["diff"], dim=1)
            for i, (md, ind) in enumerate(zip(max_diff, index)):
                logger.tb_add_scalar('diff/q%d_max'%i, md, self.num_train_steps)
                log_frame(frame, ind, 'q%d_maxdiff/cur_obs'%i, self.num_train_steps)
                log_frame(next_frame, ind, 'q%d_maxdiff/next_obs'%i, self.num_train_steps)
            min_diff, index = torch.min(self.debug_info["diff"], dim=1)
            for i, (md, ind) in enumerate(zip(min_diff, index)):
                logger.tb_add_scalar('diff/q%d_min'%i, md, self.num_train_steps)
                log_frame(frame, ind, 'q%d_mindiff/cur_obs'%i, self.num_train_steps)
                log_frame(next_frame, ind, 'q%d_mindiff/next_obs'%i, self.num_train_steps)
            logger.tb_flush()

    def train_from_torch_batch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        cur_frames = batch['observations']
        actions = batch['actions']
        next_frames = batch['next_observations']
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
        #process frames
        obs_aug = self.processor(frame_aug)
        if self.num_train_steps % 100 == 0:
            feature_map = self.processor.feature_map(frame_aug)
            log_feature_map(feature_map.mean(1),frame_aug,0,'',self.num_train_steps)
        actor_next_obs_aug = self.processor(next_frame_aug)
        if self.detach_step & self.num_train_steps>self.detach_step:
            obs_aug = obs_aug.detach()
            actor_next_obs_aug = actor_next_obs_aug.detach()            
        if self.target_processor is None:
            next_obs_aug = actor_next_obs_aug
        else:
            next_obs_aug = self.target_processor(next_frame_aug)
        if self.analyze_embeddings:
            e, next_e = obs_aug[batch_size:], actor_next_obs_aug[batch_size:]
            s, next_s = batch['states'], batch['next_states']
            projection_loss = self._update_projection(e, next_e, s, next_s)
            self.train_info['embedding_to_state/loss'] = projection_loss.item()

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
        ###########################
        # update actor and alpha #
        ###########################
        if self.num_train_steps % self.policy_update_freq == 0:
            actor_obs = obs_aug[:batch_size].detach()
            new_action, action_info = self.policy.action(
                actor_obs, **self.current_sample_kwargs
            )
            # update alpha
            if self.use_automatic_entropy_tuning:
                alpha_loss, train_alpha_info = self.compute_alpha_loss(action_info)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.train_info.update(train_alpha_info)
            # update policy
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

    def _update_projection(
        self, 
        cur_e,
        next_e, 
        cur_s,
        next_s
    ):
        assert self.n_aug == 2
        batch_size = cur_e.shape[0]
        with torch.no_grad():
            e = torch.cat([cur_e, next_e], dim=0)
            s = torch.cat([cur_s, next_s], dim=0)
            #### normalize
            batch_mean = torch.mean(s, dim=0)
            batch_std = torch.std(s, dim=0) + 1e-6
            if self._s_mean is None:
                self._s_mean = batch_mean
                self._s_std = batch_std
            else:
                self._s_mean = self._s_fac*self._s_mean + (1-self._s_fac)*batch_mean
                self._s_std = self._s_fac*self._s_std + (1-self._s_fac)*batch_std
            normalized_s = (s-self._s_mean) / self._s_std
        # update
        linear_projection = self.linear_projection
        projection_optimizer = self.projection_optimizer
        normalized_pred_s = linear_projection(e.detach())
        projection_loss = F.mse_loss(normalized_pred_s, normalized_s.detach())
        projection_optimizer.zero_grad()
        projection_loss.backward()
        projection_optimizer.step()
        if self._log_tb_or_not() and logger.log_or_not(logger.INFO):
            pred_s = normalized_pred_s*self._s_std + self._s_mean
            pred_s = pred_s.view(2, batch_size, self._s_size)
            pred_s = pred_s.detach().cpu().numpy()
            s = s.view(2, batch_size, self._s_size)
            s = s.detach().cpu().numpy()
            abs_diff = np.abs(pred_s - s)
            prefix = "embedding_to_state"
            for i, name in enumerate(self._state_dim_name):
                _prefix = prefix + '-' + name
                _s = s[:,:,i]
                _pred_s = pred_s[:,:,i]
                _abs_diff = abs_diff[:,:,i]
                min_x, max_x = _s.min(), _s.max()
                fig, ax = plt.subplots()
                ax.grid(ls='--')
                ax.plot([min_x,max_x],[min_x,max_x], color='gray', ls='--', alpha=0.5)
                for t in range(2):
                    label = "cur_state" if t == 0 else "next_state"
                    ax.scatter(_s[t,:], _pred_s[t,:], alpha=0.2, label=label)
                    _fig, _ax = plt.subplots()
                    _ax.grid(ls='--')
                    _ax.plot([min_x,max_x],[min_x,max_x], color='gray', ls='--', alpha=0.5)
                    _ax.scatter(_s[t,:], _pred_s[t,:], alpha=0.2)
                    logger.tb_add_figure("%s/%s"%(_prefix, label), _fig, self.num_train_steps)
                ax.legend()
                logger.tb_add_figure("%s/all"%_prefix, fig, self.num_train_steps)
                _prefix = prefix + '/' + name
                logger.tb_add_scalar("%s/abs_diff_scalar"%_prefix, _abs_diff[:,:,].mean(), self.num_train_steps)
                logger.tb_add_histogram("%s/abs_diff_histogram"%_prefix, _abs_diff, self.num_train_steps)
                logger.tb_flush()
        return projection_loss

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
            self.plot_value_scatter('anomaly/')
            self.anomaly_count += 1
            if self.anomaly_count % 10000 == 0 :
                pdb.set_trace()
            anomaly = True
        else:
            anomaly = False
        info_keys = list(self.debug_info.keys())
        for k in info_keys:
            del self.debug_info[k]
        return anomaly
        

