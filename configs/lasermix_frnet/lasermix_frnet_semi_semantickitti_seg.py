_base_ = [
    '../_base_/datasets/semi_semantickitti_seg.py', 
    #'../_base_/models/frnet.py',
    '../_base_/schedules/schedule.py', 
    '../_base_/default_runtime.py'
]



custom_imports = dict(
    imports=['frnet.datasets', 'frnet.datasets.transforms', 'frnet.models'],
    allow_failed_imports=False)

"""
datasets : NuScenesSegDataset  (SemanticKITTI 는 필요 없음)
datasets.transforms : FrustumMix, RangeInterpolation, InstanceCopy
models : FRNetBackbone, FrustumRangePreprocessor, FRHead, FrustumHead, BoundaryLoss, FRNet, FrustumFeatureEncoder
"""


segmentor = dict(
    type='FRNet',
    data_preprocessor=dict(type='FrustumRangePreprocessor',
                           H=64, W=512, fov_up=3.0,fov_down=-25.0, ignore_index=19),
    voxel_encoder=dict(
        type='FrustumFeatureEncoder',
        in_channels=4,
        feat_channels=(64, 128, 256, 256),
        with_distance=True,
        with_cluster_center=True,
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        with_pre_norm=True,
        feat_compression=16),
    backbone=dict(
        type='FRNetBackbone',
        in_channels=16,
        point_in_channels=384,
        output_shape=(64, 512),    # frnet 코드 참조하여 추가
        depth=34,
        stem_channels=128,
        num_stages=4,
        out_channels=(128, 128, 128, 128),
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        fuse_channels=(256, 128),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        point_norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='HSwish', inplace=True)),
    decode_head=dict(
        type='FRHead',
        in_channels=128,
        middle_channels=(128, 256, 128, 64),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        channels=64,
        num_classes=20,    # frnet 코드 참조하여 추가
        ignore_index=19,   # frnet 코드 참조하여 추가
        dropout_ratio=0,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0),
        conv_seg_kernel_size=1),
    auxiliary_head=[             # frnet 코드 참조하여 추가
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19),
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19,
            indices=2),
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19,
            indices=3),
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19,
            indices=4),
    ])

model = dict(
    type='LaserMix', 
    segmentor_student=segmentor, 
    segmentor_teacher=segmentor,
    data_preprocessor=dict(
        type='MultiBranch3DDataPreprocessor',
        data_preprocessor=dict(H=64, W=512, fov_up=3.0, fov_down=-25.0, ignore_index=19),   #FRNet 전처리 사용
    ),

    loss_mse=(dict(type='mmdet.MSELoss', loss_weight=250)),
    semi_train_cfg=dict(
        freeze_teacher=True, pseudo_thr=0.9, ignore_label=19,
        pitch_angles=[-25, 3], num_areas=[4, 5, 6, 7, 8],
        sup_weight=1, unsup_weight=1,
    ),
    semi_test_cfg=dict(extract_feat_on='teacher', predict_on='teacher'))


# EMA 가중치 업데이트 방식, mmdet에 추가가 되어 있지 않음 lasermix.py에 직접적으로 구현되어 있는것으로 확인된다.
# custom_hooks = [dict(type='mmdet.MeanTeacherHook', momentum=0.01)]  