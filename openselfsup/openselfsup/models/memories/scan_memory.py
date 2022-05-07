import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.distributed as dist
from mmcv.runner import get_dist_info

from ..registry import MEMORIES


@MEMORIES.register_module
class SCANMemory(nn.Module):
    """Memory modules for SCAN.

    Args:
        length (int): Number of features stored in samples memory.
        feat_dim (int): Dimension of stored features.
        momentum (float): Momentum coefficient for updating features.
        num_classes (int): Number of classes.
        num_clusters (int): Number of clusters.
        min_cluster (int): Minimal cluster size.
    """

    def __init__(self, length, feat_dim, momentum, num_classes, num_clusters, min_cluster,
                 **kwargs):
        super(SCANMemory, self).__init__()
        self.rank, self.num_replicas = get_dist_info()

        if self.rank == 0: #remove attribute error
            self.feature_bank = torch.zeros((length, feat_dim),
                                            dtype=torch.float32)
        self.cluster_label_bank = torch.zeros((length, ), dtype=torch.long)
        self.class_label_bank = torch.zeros((length, ), dtype=torch.long)
        self.cluster_centroids = torch.zeros((num_clusters, feat_dim),
                                     dtype=torch.float32).cuda()
        self.class_centroids = torch.zeros((num_classes, feat_dim),
                                     dtype=torch.float32).cuda()                   
        self.kmeans = KMeans(n_clusters=2, random_state=0, max_iter=20)
        self.feat_dim = feat_dim
        self.initialized = False
        self.momentum = momentum
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.min_cluster = min_cluster
        self.debug = kwargs.get('debug', False)

    def init_memory(self, feature, p_label, gt_label):
        """Initialize memory modules."""
        self.initialized = True
        self.cluster_label_bank.copy_(torch.from_numpy(p_label).long())
        self.class_label_bank.copy_(torch.from_numpy(gt_label).long())
        # make sure no empty clusters
        assert (np.bincount(p_label, minlength=self.num_clusters) != 0).all()
        if self.rank == 0:
            feature /= (np.linalg.norm(feature, axis=1).reshape(-1, 1) + 1e-10) # L2-norm
            self.feature_bank.copy_(torch.from_numpy(feature))
            cluster_centroids = self._compute_cluster_centroids()
            self.cluster_centroids.copy_(cluster_centroids)
            class_centroids = self._compute_class_centroids()
            self.class_centroids.copy_(class_centroids)
        dist.broadcast(self.cluster_centroids, 0)

    def _compute_cluster_centroids_ind(self, cinds):
        """Compute a few centroids."""
        assert self.rank == 0
        num = len(cinds)
        cluster_centroids = torch.zeros((num, self.feat_dim), dtype=torch.float32)
        for i, c in enumerate(cinds):
            ind = np.where(self.cluster_label_bank.numpy() == c)[0]
            cluster_centroids[i, :] = self.feature_bank[ind, :].mean(dim=0)
        return cluster_centroids

    def _compute_cluster_centroids(self):
        """Compute all non-empty cluster centroids."""
        assert self.rank == 0
        l = self.cluster_label_bank.numpy()
        argl = np.argsort(l)
        sortl = l[argl]
        diff_pos = np.where(sortl[1:] - sortl[:-1] != 0)[0] + 1
        start = np.insert(diff_pos, 0, 0)
        end = np.insert(diff_pos, len(diff_pos), len(l))
        class_start = sortl[start]
        # keep empty class centroids unchanged
        cluster_centroids = self.cluster_centroids.cpu().clone()
        for i, st, ed in zip(class_start, start, end):
            cluster_centroids[i, :] = self.feature_bank[argl[st:ed], :].mean(dim=0)
        return cluster_centroids

    def _compute_class_centroids(self):
        """Compute all non-empty class centroids."""
        assert self.rank == 0
        l = self.class_label_bank.numpy()
        argl = np.argsort(l)
        sortl = l[argl]
        diff_pos = np.where(sortl[1:] - sortl[:-1] != 0)[0] + 1
        start = np.insert(diff_pos, 0, 0)
        end = np.insert(diff_pos, len(diff_pos), len(l))
        class_start = sortl[start]
        # keep empty class centroids unchanged
        class_centroids = self.class_centroids.cpu().clone()
        for i, st, ed in zip(class_start, start, end):
            class_centroids[i, :] = self.feature_bank[argl[st:ed], :].mean(dim=0)
        return class_centroids

    def _gather(self, ind, feature):
        """Gather indices and features."""
        # if not hasattr(self, 'ind_gathered'):
        #    self.ind_gathered = [torch.ones_like(ind).cuda()
        #                         for _ in range(self.num_replicas)]
        # if not hasattr(self, 'feature_gathered'):
        #    self.feature_gathered = [torch.ones_like(feature).cuda()
        #                             for _ in range(self.num_replicas)]
        ind_gathered = [
            torch.ones_like(ind).cuda() for _ in range(self.num_replicas)
        ]
        feature_gathered = [
            torch.ones_like(feature).cuda() for _ in range(self.num_replicas)
        ]
        dist.all_gather(ind_gathered, ind)
        dist.all_gather(feature_gathered, feature)
        ind_gathered = torch.cat(ind_gathered, dim=0)
        feature_gathered = torch.cat(feature_gathered, dim=0)
        return ind_gathered, feature_gathered

    def update_samples_memory(self, ind, feature):
        """Update samples memory."""
        assert self.initialized
        feature_norm = feature / (feature.norm(dim=1).view(-1, 1) + 1e-10)  # normalize
        ind, feature_norm = self._gather(
            ind, feature_norm)  # ind: (N*w), feature: (N*w)xk, cuda tensor
        ind = ind.cpu()

        if self.rank == 0:
            feature_old = self.feature_bank[ind, ...].cuda()
            feature_new = (1 - self.momentum) * feature_old + self.momentum * feature_norm
            feature_norm = feature_new / (
                feature_new.norm(dim=1).view(-1, 1) + 1e-10)
            self.feature_bank[ind, ...] = feature_norm.cpu()
            
        dist.barrier()
        dist.broadcast(feature_norm, 0)
        # compute new labels
        similarity_to_cluster_centroids = torch.mm(self.cluster_centroids,
                                           feature_norm.permute(1, 0))  # CxN
        newlabel = similarity_to_cluster_centroids.argmax(dim=0)  # cuda tensor
        newlabel_cpu = newlabel.cpu()
        change_ratio = (newlabel_cpu !=
            self.cluster_label_bank[ind]).sum().float().cuda() \
            / float(newlabel_cpu.shape[0])
        self.cluster_label_bank[ind] = newlabel_cpu.clone()  # copy to cpu
        return change_ratio

    def deal_with_small_clusters(self):
        """Deal with small clusters."""
        # check empty class
        hist = np.bincount(self.cluster_label_bank.numpy(), minlength=self.num_clusters)
        small_clusters = np.where(hist < self.min_cluster)[0].tolist()
        if self.debug and self.rank == 0:
            print("mincluster: {}, num of small class: {}".format(
                hist.min(), len(small_clusters)))
        if len(small_clusters) == 0:
            return
        # re-assign samples in small clusters to make them empty
        for s in small_clusters:
            ind = np.where(self.cluster_label_bank.numpy() == s)[0]
            if len(ind) > 0:
                inclusion = torch.from_numpy(
                    np.setdiff1d(
                        np.arange(self.num_clusters),
                        np.array(small_clusters),
                        assume_unique=True)).cuda()
                if self.rank == 0:
                    target_ind = torch.mm(
                        self.cluster_centroids[inclusion, :],
                        self.feature_bank[ind, :].cuda().permute(
                            1, 0)).argmax(dim=0)
                    target = inclusion[target_ind]
                else:
                    target = torch.zeros((ind.shape[0], ),
                                         dtype=torch.int64).cuda()
                dist.all_reduce(target)
                self.cluster_label_bank[ind] = torch.from_numpy(target.cpu().numpy())
        # deal with empty cluster
        self._redirect_empty_clusters(small_clusters)

    def update_cluster_centroids_memory(self, cinds=None):
        """Update centroids memory."""
        if self.rank == 0:
            if self.debug:
                print("updating cluster_centroids ...")
            if cinds is None:
                center = self._compute_cluster_centroids()
                self.cluster_centroids.copy_(center)
            else:
                center = self._compute_cluster_centroids_ind(cinds)
                self.cluster_centroids[
                    torch.LongTensor(cinds).cuda(), :] = center.cuda()
        dist.broadcast(self.cluster_centroids, 0)
    
    def update_class_centroids_memory(self):
        """Update class centroids memory."""
        if self.rank == 0:
            if self.debug:
                print("updating cluster_centroids ...")
            center = self._compute_class_centroids()
            self.class_centroids.copy_(center)  
        dist.broadcast(self.class_centroids, 0)

    def _partition_max_cluster(self, max_cluster):
        """Partition the largest cluster into two sub-clusters."""
        assert self.rank == 0
        max_cluster_inds = np.where(self.cluster_label_bank == max_cluster)[0]

        assert len(max_cluster_inds) >= 2
        max_cluster_features = self.feature_bank[max_cluster_inds, :]
        if np.any(np.isnan(max_cluster_features.numpy())):
            raise Exception("Has nan in features.")
        kmeans_ret = self.kmeans.fit(max_cluster_features)
        sub_cluster1_ind = max_cluster_inds[kmeans_ret.labels_ == 0]
        sub_cluster2_ind = max_cluster_inds[kmeans_ret.labels_ == 1]
        if not (len(sub_cluster1_ind) > 0 and len(sub_cluster2_ind) > 0):
            print(
                "Warning: kmeans partition fails, resort to random partition.")
            sub_cluster1_ind = np.random.choice(
                max_cluster_inds, len(max_cluster_inds) // 2, replace=False)
            sub_cluster2_ind = np.setdiff1d(
                max_cluster_inds, sub_cluster1_ind, assume_unique=True)
        return sub_cluster1_ind, sub_cluster2_ind

    def _redirect_empty_clusters(self, empty_clusters):
        """Re-direct empty clusters."""
        for e in empty_clusters:
            assert (self.cluster_label_bank != e).all().item(), \
                "Cluster #{} is not an empty cluster.".format(e)
            
            max_cluster = np.bincount(
                self.cluster_label_bank, minlength=self.num_clusters).argmax().item()
            # gather partitioning indices
            if self.rank == 0:
                sub_cluster1_ind, sub_cluster2_ind = self._partition_max_cluster(
                    max_cluster)
                size1 = torch.LongTensor([len(sub_cluster1_ind)]).cuda()
                size2 = torch.LongTensor([len(sub_cluster2_ind)]).cuda()
                sub_cluster1_ind_tensor = torch.from_numpy(
                    sub_cluster1_ind).long().cuda()
                sub_cluster2_ind_tensor = torch.from_numpy(
                    sub_cluster2_ind).long().cuda()
            else:
                size1 = torch.LongTensor([0]).cuda()
                size2 = torch.LongTensor([0]).cuda()
            dist.all_reduce(size1)
            dist.all_reduce(size2)
            if self.rank != 0:
                sub_cluster1_ind_tensor = torch.zeros(
                    (size1, ), dtype=torch.int64).cuda()
                sub_cluster2_ind_tensor = torch.zeros(
                    (size2, ), dtype=torch.int64).cuda()
            dist.broadcast(sub_cluster1_ind_tensor, 0)
            dist.broadcast(sub_cluster2_ind_tensor, 0)
            if self.rank != 0:
                sub_cluster1_ind = sub_cluster1_ind_tensor.cpu().numpy()
                sub_cluster2_ind = sub_cluster2_ind_tensor.cpu().numpy()

            # reassign samples in partition #2 to the empty class
            self.cluster_label_bank[sub_cluster2_ind] = e
            # update centroids of max_cluster and e
            self.update_cluster_centroids_memory([max_cluster, e])

    def get_triplet_pos_and_neg(self, gt_label, ind, features):
        """retrun positive and negative samples for calculating purity loss"""
        positives = self.class_centroids[gt_label]
        cluster_idx = self.cluster_label_bank[ind]
        negatives = []

        for i, feat in enumerate(features):
            cluster_members_idx = [j for j, (e,f) in enumerate(zip(self.cluster_label_bank,self.class_label_bank)) if e == cluster_idx[i] and f != gt_label[i]]
            if not cluster_members_idx:
                # deal with purity cluster
                cluster_members_idx = [j for j, f in enumerate(self.class_label_bank) if f != gt_label[i]]
            cluster_members_feats = self.feature_bank[cluster_members_idx].cuda()

            # the euclidean distance between feature[i] and thoes features
            distance_i = torch.cdist(feat.unsqueeze(0), cluster_members_feats, p=2).squeeze()
            # the indx of sample with minimum distance
            negative_idx = cluster_members_idx[torch.argmin(distance_i)]
            # append the feature of this sample
            negatives.append(self.feature_bank[negative_idx])

        negatives = torch.stack(negatives, dim=0)

        return positives, negatives
