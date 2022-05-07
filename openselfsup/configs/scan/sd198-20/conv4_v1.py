_base_ = '../../base.py'
# model settings
num_classes = 200
num_clusters = 50
model = dict(
    type='SCAN',
    pretrained=None,
    with_sobel=False,
    backbone=dict(
        type='ConvNet',
        depth=4),
    neck=dict(
        type='NonLinearNeckV0',
        in_channels=1600,
        hid_channels=512,
        out_channels=256,
        with_avg_pool=False),
    head=dict(
        type='ClusterHead3Losses',
        with_avg_pool=False,
        in_channels=256,
        num_classes=num_classes,
        num_clusters=num_clusters),
    memory_bank=dict(
        type='SCANMemory',
        length=1200,
        feat_dim=256,
        momentum=0.5,
        num_classes=num_classes,
        num_clusters=num_clusters,
        min_cluster=15,
        debug=False))
# dataset settings
data_source_cfg = dict(
    type='SD198',
    memcached=False,
    mclient_path=None)
data_train_list = '../SkinLesionData/SD-198-20/base.json'
data_train_root = '../SkinLesionData/SD-198-20'
data_test_list = '../SkinLesionData/SD-198-20/base.json'
data_test_root = '../SkinLesionData/SD-198-20'
dataset_type = 'SCANDataset'

img_norm_cfg = dict(mean=[0.611, 0.485, 0.449], std=[0.215, 0.209, 0.212]) # SD-198-20 base.json
jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)
train_pipeline = [
    dict(type='RandomResizedCrop', size=80),
    dict(type='ImageJitter', transformdict=jitter_param),
    dict(type='RandomRotation', degrees=30),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=92),
    dict(type='CenterCrop', size=80),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
extract_pipeline = [
    dict(type='Resize', size=92),
    dict(type='CenterCrop', size=80),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]

prefetch = False
data = dict(
    imgs_per_gpu=16,  
    sampling_replace=True,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline=test_pipeline))
# additional hooks
custom_hooks = [
    dict(
        type='DeepClusterHookScan',
        extractor=dict(
            imgs_per_gpu=16,
            workers_per_gpu=2,
            dataset=dict(
                type=dataset_type,
                data_source=dict(
                    list_file=data_train_list,
                    root=data_train_root,
                    **data_source_cfg),
                pipeline=extract_pipeline),
            return_gt_label=True),
        clustering=dict(type='Kmeans', k=num_clusters, pca_dim=-1),  # no pca
        unif_sampling=False,
        reweight=True,
        reweight_pow=0.5,
        init_memory=True,
        initial=True,  # call initially
        interval=9999999999),  # initial only
    dict(
        type='ODCHookScan',
        cluster_centroids_update_interval=10,  # iter
        class_centroids_update_interval=1,  # iter
        deal_with_small_clusters_interval=1,
        evaluate_interval=50,
        reweight=True,
        reweight_pow=0.5),
    dict(
        type='ValidateHook',
        dataset=data['val'],
        initial=True,
        interval=1,
        imgs_per_gpu=32,
        workers_per_gpu=4,
        prefetch=prefetch,
        img_norm_cfg=img_norm_cfg,
        eval_param=dict(topk=(1,)))
]

# optimizer
optimizer = dict(
    type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.00001,
    nesterov=False,
    paramwise_options={'\Ahead.': dict(momentum=0.)})
# learning policy
lr_config = dict(policy='step', step=[800], gamma=0.4)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 800

# CUDA_VISIBLE_DEVICES=3 bash tools/dist_train.sh configs/scan/sd198-20/conv4_v1.py 1
