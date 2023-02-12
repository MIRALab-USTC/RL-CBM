from cbm.processors.base_processor import Processor
from cbm.torch_modules.mlp import MLP
import cbm.torch_modules.utils as ptu
from cbm.utils.logger import logger
from cbm.agents.ddpg_agent import plot_scatter
from cbm.agents.drq.drq_agent import log_frame
import numpy as np
from torch import nn
import torch

def sinkhorn(scores, eps=0.05, n_iter=3):
    def remove_infs(x):
        m = x[torch.isfinite(x)].max().item()
        x[torch.isinf(x)] = m
        x[x==0] = 1e-38
        return x
    B, K = scores.shape
    scores = scores.view(B*K)
    Q = torch.softmax(scores/eps, dim=0)
    Q = remove_infs(Q).view(B,K).T
    r, c = ptu.ones(K)/K, ptu.ones(B)/B
    for _ in range(n_iter):
        u = (r/torch.sum(Q, dim=1))
        Q *= remove_infs(u).unsqueeze(1)
        v = (c/torch.sum(Q,dim=0))
        Q *= remove_infs(v).unsqueeze(0)
    bsum = torch.sum(Q,dim=0,keepdim=True)
    output = ( Q / remove_infs(bsum)).T
    assert torch.isnan(output.sum())==False
    return output

def compute_cl_loss(e1, e2, alpha):
    similarity = compute_cosine_similarity(e1, e2)
    similarity = similarity/alpha
    with torch.no_grad():
        pred_prob = torch.softmax(similarity, dim=-1)
        target_prob = ptu.eye(len(similarity))
        accuracy = (pred_prob * target_prob).sum(-1)
        diff = pred_prob-target_prob
    loss = (similarity*diff).sum(-1).mean()
    return loss, pred_prob, accuracy

def compute_cosine_similarity(e1, e2):
    e1_norm = torch.norm(e1, dim=-1, p=2, keepdim=True) 
    e1 = e1 / e1_norm
    e2_norm = torch.norm(e2, dim=-1, p=2, keepdim=True) 
    e2 = e2 / e2_norm
    similarity = torch.mm(e1, torch.t(e2))
    return similarity

def compute_l2_similarity(e1, e2, bound=2):
    e1 = e1[:,None,:]
    e2 = e2[None,:,:]
    diff = e1 - e2
    l2_sim = -(diff**2).mean(-1)
    l2_sim = torch.clamp(l2_sim, -bound, bound)
    return l2_sim

def compute_abs_similarity(e1, e2, bound=2):
    e1 = e1[:,None,:]
    e2 = e2[None,:,:]
    diff = e1 - e2
    abs_sim = -torch.abs(diff).mean(-1)
    return abs_sim

class ProtoModelEMA(nn.Module, Processor):
    def __init__(
        self,
        env,
        processor,
        policy,
        embedding_size=50,
        proto_size=128, 
        forward_layers=[256,256],
        z_coef=1,
        k=128,
        alpha=0.1,
        activation='relu',
        cl_temp=0.1,
        cluster_next_coef=1,
        cluster_r_coef=1,
        cluster_coef=1,
        r_seq = 3,
        diff_mode = 'l2',
        tau = 0.99,
        **proto_kwargs
    ):
        nn.Module.__init__(self)
        self.embedding_size = embedding_size
        self.action_size = env.action_shape[0]
        self.trunk = nn.Sequential(
            nn.Linear(processor.output_shape[0], self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.Tanh())
        self.policy = policy
        assert diff_mode == "l2" or "abs"
        if diff_mode == "l2":
            self.rew_diff = compute_l2_similarity
        elif diff_mode == "abs":
            self.rew_diff = compute_abs_similarity
        self.forward_layers = forward_layers
        self.activation = activation       
        self.forward_predict_net = MLP(
            embedding_size + self.action_size,
            embedding_size + 1, # next_proto, r
            hidden_layers=forward_layers,
            activation=activation)
        
        self.forward_trunk = nn.Sequential(
            nn.LayerNorm(self.embedding_size),
            nn.Tanh())
        self.k = k
        self.proto_size = proto_size
        self.k_proto = nn.Parameter(ptu.zeros(k, self.embedding_size))
        self.k_proto.requires_grad_(True)
        self.alpha = alpha
        self.z_coef = z_coef
        self.cl_temp = cl_temp
        self.cluster_next_coef = cluster_next_coef
        self.cluster_r_coef = cluster_r_coef
        self.cluster_coef = cluster_coef
        self.has_init = False
        self.action_repeat = env.action_repeat
        self.r_seq = r_seq
        input_size = (self.embedding_size + self.action_size)
        self.transition = MLP(
            input_size,
            self.embedding_size, #next_proto
            hidden_layers=self.forward_layers,
            activation=self.activation)
        self.forward_trunk = nn.Sequential(
            nn.LayerNorm(self.embedding_size),
            nn.Tanh())
        self.proto_r = ptu.zeros(self.k,1)
        self.has_r_init = False
        self.tau = tau
    
    def init_k_proto(self, z_batch):
        from cbm.torch_modules.utils import device
        k_proto = nn.Linear(self.embedding_size, self.k).weight.data.to(device)
        self.k_proto.data = k_proto

    def ema_proto_r(self, r, prob, batch_size):
        new_r = torch.mm(torch.t(prob),r)/(batch_size/self.k) 
        if self.has_r_init == 0:
            self.proto_r = new_r
            self.has_r_init = True
        else:
            self.proto_r = self.tau * self.proto_r + (1-self.tau)*new_r

    def _compute_auxiliary_loss(self, next_z, tar_next_z):
        if self.z_coef>0:
            h_loss, _, _ = compute_cl_loss(next_z, tar_next_z, self.cl_temp)
        else: 
            h_loss = 0
        loss = self.z_coef*h_loss  
        return loss, h_loss

    def compute_auxiliary_loss(self, obs, a, r ,next_obs, next_a, 
            n_step=0, log=False, next_frame=None):
        r = r/self.action_repeat
        batch_size = len(obs)

        # model 
        cat_obs = torch.cat([obs, next_obs],dim=0)
        cat_z = self.trunk(cat_obs)
        z, next_z = torch.chunk(cat_z, 2)
        if not self.has_init:
            self.init_k_proto(next_z)
            self.has_init = True
        feature = torch.cat([z,a],dim=-1)
        pred_z = self.transition(feature)
        pred_next_z = self.forward_trunk(pred_z)
        model_loss, h_loss =self._compute_auxiliary_loss( 
            pred_next_z, next_z.detach())

        # cluster
        if self.cluster_coef > 0:
            with torch.no_grad():
                with self.policy.deterministic_(True):
                    proto_action, _ = self.policy.action_from_feature(self.k_proto)
            proto_feature = torch.cat([self.k_proto, proto_action],dim=-1)

            proto_z = self.transition(proto_feature)
            proto_next_z = self.forward_trunk(proto_z)
            cur_z = z
            z_sim = compute_cosine_similarity(cur_z, proto_z)
            with torch.no_grad():
                pred_prob = torch.softmax(z_sim/self.alpha,dim=-1)
                next_z_sim = compute_cosine_similarity(next_z, proto_next_z)
                r_sim = self.rew_diff(r, self.proto_r) 
                sim = self.cluster_r_coef*r_sim + self.cluster_next_coef*next_z_sim
                tar_prob = sinkhorn(sim)
                self.ema_proto_r(r, tar_prob, batch_size)
                pred_diff = pred_prob - tar_prob
            cl_loss = (pred_diff*z_sim).sum(-1).mean()
            cluster_loss = cl_loss
        else:
            cluster_loss = 0
        
        loss = model_loss + self.cluster_coef*cluster_loss
        
        # log
        if log and n_step%100==0:
            ## model
            logger.tb_add_scalar("model/h_loss", h_loss, n_step)
            with torch.no_grad():
                z_diff = pred_next_z - next_z
                z_diff_norm = torch.norm(z_diff,dim=-1,p=2)
                shuffle_ind = np.random.randint(batch_size,size=(batch_size,))
                shuffle_next_h = next_z[shuffle_ind]
                shuffle_diff = pred_next_z - shuffle_next_h
                shuffle_diff_norm = torch.norm(shuffle_diff,dim=-1,p=2)
            plot_scatter(z_diff_norm, shuffle_diff_norm, "no", "model/preddiff_vs_shufflediff", n_step)
            ## cluster
            logger.tb_add_scalar("cluster/cl_loss", cl_loss, n_step)
            logger.tb_add_scalar("cluster/cluster_loss", cluster_loss, n_step)
            logger.tb_add_histogram("cluster/proto_r",self.proto_r,n_step)
            if n_step%5000 ==0:
                plot_scatter(tar_prob, pred_prob, "x=y", "cluster/tar_vs_pred", n_step)
                plot_scatter(sim, z_sim, "no", "cluster/sim_vs_zsim", n_step)
            if next_frame is not None and n_step%10000 ==0:
                sim_cluster0 = z_sim[:,0]
                _,ind = torch.topk(sim_cluster0,3)
                for i, topi in enumerate(ind):
                    log_frame(next_frame, topi, "c0_top%d"%i, n_step, "cluster/")
                sim_cluster1 = z_sim[:,1]
                _,ind = torch.topk(sim_cluster1,3)
                for i, topi in enumerate(ind):
                    log_frame(next_frame, topi, "c1_top%d"%i, n_step, "cluster/")

        return loss

    def process(self):
        raise NotImplementedError
