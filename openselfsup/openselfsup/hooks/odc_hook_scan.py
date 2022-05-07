import numpy as np

from mmcv.runner import Hook

from openselfsup.utils import print_log
from .registry import HOOKS


@HOOKS.register_module
class ODCHookScan(Hook):
    """Hook for SCAN.

    Args:
        cluster_centroids_update_interval (int): Frequency of iterations
            to update cluster centroids.
        class_centroids_update_interval (int): Frequency of iterations
            to update class centroids.
        deal_with_small_clusters_interval (int): Frequency of iterations
            to deal with small clusters.
        evaluate_interval (int): Frequency of iterations to evaluate clusters.
        reweight (bool): Whether to perform loss re-weighting.
        reweight_pow (float): The power of re-weighting.
        dist_mode (bool): Use distributed training or not. Default: True.
    """

    def __init__(self,
                 cluster_centroids_update_interval,
                 class_centroids_update_interval,
                 deal_with_small_clusters_interval,
                 evaluate_interval,
                 reweight,
                 reweight_pow,
                 dist_mode=True):
        assert dist_mode, "non-dist mode is not implemented"
        self.cluster_centroids_update_interval = cluster_centroids_update_interval
        self.class_centroids_update_interval = class_centroids_update_interval
        self.deal_with_small_clusters_interval = \
            deal_with_small_clusters_interval
        self.evaluate_interval = evaluate_interval
        self.reweight = reweight
        self.reweight_pow = reweight_pow

    def after_train_iter(self, runner):
        # cluster centroids update
        if self.every_n_iters(runner, self.cluster_centroids_update_interval):
            runner.model.module.memory_bank.update_cluster_centroids_memory()

        # class centroids update
        if self.every_n_iters(runner, self.class_centroids_update_interval):
            runner.model.module.memory_bank.update_class_centroids_memory()

        # deal with small clusters
        if self.every_n_iters(runner, self.deal_with_small_clusters_interval):
            runner.model.module.memory_bank.deal_with_small_clusters()

        # reweight
        runner.model.module.set_reweight()

        # evaluate
        if self.every_n_iters(runner, self.evaluate_interval):
            new_labels = runner.model.module.memory_bank.cluster_label_bank
            if new_labels.is_cuda:
                new_labels = new_labels.cpu()
            self.evaluate(runner, new_labels.numpy())

    def after_train_epoch(self, runner):
        # save cluster
        if self.every_n_epochs(runner, 10) and runner.rank == 0:
            new_labels = runner.model.module.memory_bank.cluster_label_bank
            if new_labels.is_cuda:
                new_labels = new_labels.cpu()
            np.save(
                "{}/cluster_epoch_{}.npy".format(runner.work_dir,
                                                 runner.epoch),
                new_labels.numpy())

    def evaluate(self, runner, new_labels):
        hist = np.bincount(
            new_labels, minlength=runner.model.module.memory_bank.num_clusters)
        empty_cls = (hist == 0).sum()
        minimal_cls_size, maximal_cls_size = hist.min(), hist.max()
        if runner.rank == 0:
            print_log(
                "empty_num: {}\tmin_cluster: {}\tmax_cluster:{}".format(
                    empty_cls.item(), minimal_cls_size.item(),
                    maximal_cls_size.item()),
                logger='root')
