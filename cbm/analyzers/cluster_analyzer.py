from collections import OrderedDict
from cbm.analyzers.base_analyzer import Analyzer
from cbm.utils.logger import logger
import torch 
import numpy as np
import os
from cbm.agents.cbm.cbm_model import compute_cosine_similarity
from cbm.agents.drq.drq_agent import log_frame

class ClusterAnalyzer(Analyzer):
    def __init__(
        self,
        agent,
        proto_model,
        pool,
        batch_size=512,
        ndomain = 0,
        log_n = 3
    ):
        self.work = proto_model.cluster_coef
        self.pool = pool
        self.proto_model = proto_model
        self.batch_size = batch_size 
        self.agent = agent
        self.log_n = log_n
        self.ndomain = ndomain

    def analyze(self, epoch): 
        if self.work > 0 and epoch>0:
            batch = self.pool.analyze_sample(self.batch_size) 
            frames = batch['frames']
            frame_stack = self.agent.frame_stack
            C, H, W = frames.shape[2], frames.shape[3], frames.shape[4]
            cur_frames = frames[:,:frame_stack].reshape(-1, C*frame_stack, H, W)
            obs = self.agent.processor(cur_frames)
            cur_z = self.proto_model.trunk(obs)
            proto_z = self.proto_model.k_proto
            z_sim = compute_cosine_similarity(cur_z,proto_z)

            sim_cluster0 = z_sim[:,0]
            _,ind = torch.topk(sim_cluster0,self.log_n)
            for i, topi in enumerate(ind):
                log_frame(cur_frames, topi, "c1_top%d"%i, epoch, "visualize/")

            sim_cluster1 = z_sim[:,1]
            _,ind = torch.topk(sim_cluster1,self.log_n)
            for i, topi in enumerate(ind):
                log_frame(cur_frames, topi, "c2_top%d"%i, epoch, "visualize/")

            sim_cluster2 = z_sim[:,2]
            _,ind = torch.topk(sim_cluster2,self.log_n)
            for i, topi in enumerate(ind):
                log_frame(cur_frames, topi, "c3_top%d"%i, epoch, "visualize/")

            sim_cluster3 = z_sim[:,3]
            _,ind = torch.topk(sim_cluster3,self.log_n)
            for i, topi in enumerate(ind):
                log_frame(cur_frames, topi, "c4_top%d"%i, epoch, "visualize/")

            sim_cluster4 = z_sim[:,4]
            _,ind = torch.topk(sim_cluster4,self.log_n)
            for i, topi in enumerate(ind):
                log_frame(cur_frames, topi, "c5_top%d"%i, epoch, "visualize/")

class ClusterSaver(Analyzer):
    def __init__(
        self,
        agent,
        proto_model,
        pool,
        batch_size=1024,
        ndomain = 0,
    ):
        self.pool = pool
        self.proto_model = proto_model
        self.batch_size = batch_size 
        self.agent = agent
        self.ndomain = ndomain

    def analyze(self, epoch): 
        batch = self.pool.analyze_sample(self.batch_size) 
        frames = batch['frames']
        frame_stack = self.agent.frame_stack
        C, H, W = frames.shape[2], frames.shape[3], frames.shape[4]
        cur_frames = frames[:,:frame_stack].reshape(-1, C*frame_stack, H, W)
        obs = self.agent.processor(cur_frames)
        cur_z = self.proto_model.trunk(obs)
        proto_z = self.proto_model.k_proto
        z_sim = compute_cosine_similarity(cur_z,proto_z)
        
        state = batch['states'][:,frame_stack-1].cpu().numpy()
        physics = batch['physics'][:,frame_stack-1].cpu().numpy()
        label = z_sim.argmax(dim=1).cpu().numpy()
        label = np.expand_dims(label,axis=1)
        dat = np.concatenate([state, physics, label.astype(int)],axis=1)
        os.makedirs(os.path.join(logger._snapshot_dir,'cluster'),exist_ok=True)
        logdir = os.path.join(logger._snapshot_dir,'cluster',f"clures_{epoch}.csv")
        
        np.savetxt(logdir,dat,delimiter=",")