import abc
import gtimer as gt

from cbm.algorithms.base_algorithm import RLAlgorithm
from cbm.utils.process import Progress, Silent, format_for_process
import os

from cbm.utils.logger import logger

class OffPolicyRLAlgorithm(RLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        num_epochs: int = 1000,
        batch_size: int = 256,
        num_train_loops_per_epoch: int = 1000,
        num_expl_steps_per_train_loop: int = 1,
        num_trains_per_train_loop: int = 1,
        num_eval_steps: int = 5000,
        eval_freq: bool = 1,
        max_path_length: int = 1000,
        min_num_steps_before_training: int = 0,
        silent: bool = False,
        record_video_freq: int = 50,
        analyze_freq: int = 1,
        save_pool_freq: int = -1,
        item_dict_config: dict = {},
    ) -> None:
        super().__init__(item_dict_config)
        self._need_snapshot.append('agent')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_eval_steps = num_eval_steps
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.max_path_length = max_path_length
        self.record_video_freq = record_video_freq

        self.progress_class = Silent if silent else Progress
        self.collected_samples = 0
        self.analyze_freq = analyze_freq
        self.eval_freq = eval_freq
        self.save_pool_freq = save_pool_freq
    
    def _sample(self, num_steps, **collect_kwargs):
        if num_steps > 0:
            self.expl_collector.collect_new_steps(
                num_steps, 
                self.max_path_length,
                **collect_kwargs
            )

    def _before_train(self):
        self._sample(self.min_num_steps_before_training, use_tqdm=True, step_mode='init')
        # self.pool.save("/home/qiz")
        # self.pool.load("/home/qiz")
        self.training_mode(True)
        self.agent.pretrain(self.pool)
        self.training_mode(False)
    
    def _end_epoch(self, epoch):
        if (
                self.analyze_freq > 0 and epoch>=0 and \
                    (
                        epoch % self.analyze_freq == 0 or \
                        epoch == self.num_epochs-1
                    ) and \
                hasattr(self, 'analyzer')
            ):
            self.analyzer.analyze(epoch)
            if hasattr(self, "analyzer2"):
                self.analyzer2.analyze(epoch)

        gt.stamp('analyze', unique=False)
        if (
                self.record_video_freq > 0 and epoch>=0 and \
                (
                    epoch % self.record_video_freq == 0 or \
                    epoch == self.num_epochs-1
                ) and \
                hasattr(self, 'video_env')
            ):
            self.video_env.set_video_name("epoch{}".format(epoch))
            logger.log("rollout to save video...")
            self.video_env.recored_video(
                self.agent, 
                max_path_length=self.max_path_length, 
                use_tqdm=True,
                step_mode='exploit'
            )
        gt.stamp('save video', unique=False)
        if (
                self.save_pool_freq > 0 and epoch>=0 and \
                (
                    epoch % self.save_pool_freq == 0 or \
                    epoch == self.num_epochs-1
                )
            ):
            pool_dir = os.path.join(logger._snapshot_dir, "epoch_%d"%epoch)
            os.makedirs(pool_dir, exist_ok=True)
            self.pool.save(pool_dir)
        gt.stamp('save pool', unique=False)

    def _train_epoch(self, epoch):
        progress = self.progress_class(self.num_train_loops_per_epoch * self.num_trains_per_train_loop)
        for _ in range(self.num_train_loops_per_epoch):
            # sample
            self._sample(self.num_expl_steps_per_train_loop, step_mode='explore')
            gt.stamp('exploration sampling', unique=False)
            # trainning
            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                progress.update()
                # train_data = self.pool.random_batch(self.batch_size)
                # gt.stamp('sample from pool', unique=False)
                # batch = ptu.np_to_pytorch_batch(train_data)
                # gt.stamp('cpu to gpu', unique=False)
                torch_batch = self.pool.random_batch_torch(self.batch_size)
                gt.stamp('sample torch batch', unique=False)
                params = self.agent.train_from_torch_batch(torch_batch)
                progress.set_description(format_for_process(params))
                gt.stamp('training', unique=False)
            self.training_mode(False)
        expl_return = self.expl_collector.get_diagnostics()['Return Mean']
        logger.tb_add_scalar("return/exploration", expl_return, epoch)
        # evaluation
        if (
            self.eval_freq > 0 and epoch>=0 and (
                epoch % self.eval_freq == 0 or \
                epoch == self.num_epochs-1
            ) and hasattr(self, 'eval_collector')
        ):
            self.eval_collector.collect_new_steps(
                self.num_eval_steps,
                self.max_path_length,
                step_mode='exploit'
            )
            eval_return = self.eval_collector.get_diagnostics()['Return Mean']
            logger.tb_add_scalar("return/evaluation", eval_return, epoch)
        logger.tb_flush()
        gt.stamp('evaluation sampling')
        progress.close()

class OffPolicyRLAlgorithmCarla(OffPolicyRLAlgorithm):
    def _train_epoch(self, epoch):
        progress = self.progress_class(self.num_train_loops_per_epoch * self.num_trains_per_train_loop)
        for _ in range(self.num_train_loops_per_epoch):
            # sample
            self._sample(self.num_expl_steps_per_train_loop, step_mode='explore')
            gt.stamp('exploration sampling', unique=False)
            # trainning
            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                progress.update()
                # train_data = self.pool.random_batch(self.batch_size)
                # gt.stamp('sample from pool', unique=False)
                # batch = ptu.np_to_pytorch_batch(train_data)
                # gt.stamp('cpu to gpu', unique=False)
                torch_batch = self.pool.random_batch_torch(self.batch_size)
                gt.stamp('sample torch batch', unique=False)
                params = self.agent.train_from_torch_batch(torch_batch)
                progress.set_description(format_for_process(params))
                gt.stamp('training', unique=False)
            self.training_mode(False)
        expl_return = self.expl_collector.get_diagnostics()['Return Mean']
        logger.tb_add_scalar("return/exploration", expl_return, epoch)
        # evaluation
        if (
            self.eval_freq > 0 and epoch>=0 and (
                epoch % self.eval_freq == 0 or \
                epoch == self.num_epochs-1
            ) and hasattr(self, 'eval_collector')
        ):
            self.eval_collector.collect_new_steps(
                self.num_eval_steps,
                self.max_path_length,
                action_when_terminal='stop',
                step_mode='exploit'
            )
            eval_return = self.eval_collector.get_diagnostics()['Return Mean']
            logger.tb_add_scalar("return/evaluation", eval_return, epoch)
        logger.tb_flush()
        gt.stamp('evaluation sampling')
        progress.close()