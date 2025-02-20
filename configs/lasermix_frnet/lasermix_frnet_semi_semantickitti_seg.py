_base_ = [
    '../_base_/datasets/semi_semantickitti_seg.py', '../_base_/models/frnet.py',
    '../_base_/schedules/schedule.py', '../_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['frnet.datasets', 'frnet.datasets.transforms', 'frnet.models'],
    allow_failed_imports=False)

dataset_type = 'SemanticKittiDataset'
data_root = 'data/semantickitti/'

class_names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist',
    'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
    'fence', 'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign'
]
labels_map = {
    0: 19,   # "unlabeled"
    1: 19,   # "outlier" mapped to "unlabeled" --------------mapped
    10: 0,   # "car"
    11: 1,   # "bicycle"
    13: 4,   # "bus" mapped to "other-vehicle" --------------mapped
    15: 2,   # "motorcycle"
    16: 4,   # "on-rails" mapped to "other-vehicle" ---------mapped
    18: 3,   # "truck"
    20: 4,   # "other-vehicle"
    30: 5,   # "person"
    31: 6,   # "bicyclist"
    32: 7,   # "motorcyclist"
    40: 8,   # "road"
    44: 9,   # "parking"
    48: 10,  # "sidewalk"
    49: 11,  # "other-ground"
    50: 12,  # "building"
    51: 13,  # "fence"
    52: 19,  # "other-structure" mapped to "unlabeled" ------mapped
    60: 8,   # "lane-marking" to "road" ---------------------mapped
    70: 14,  # "vegetation"
    71: 15,  # "trunk"
    72: 16,  # "terrain"
    80: 17,  # "pole"
    81: 18,  # "traffic-sign"
    99: 19,  # "other-object" to "unlabeled" ----------------mapped
    252: 0,  # "moving-car" to "car" ------------------------mapped
    253: 6,  # "moving-bicyclist" to "bicyclist" ------------mapped
    254: 5,  # "moving-person" to "person" ------------------mapped
    255: 7,  # "moving-motorcyclist" to "motorcyclist" ------mapped
    256: 4,  # "moving-on-rails" mapped to "other-vehic------mapped
    257: 4,  # "moving-bus" mapped to "other-vehicle" -------mapped
    258: 3,  # "moving-truck" to "truck" --------------------mapped
    259: 4   # "moving-other"-vehicle to "other-vehicle"-----mapped
}
"""
현재 반지도학습을 위해 lasermix 정보를 가지고 왔음 cylinder3d를 frnet으로 변경 적용
라벨링 데이터 10% 기준으로 작성되어있므로 향후 기준을 바꿔서 진행 할수 있도록 한다.
"""



metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=259
)

input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None
branch_field = ['sup', 'unsup']

randomness = dict(seed=1205, deterministic=False, diff_rank_seed=True)

# pipeline used to augment labeled data,ㄴ
# which will be sent to student model for supervised training.
pre_transform = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.1415926, 3.1415926],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1])
]

sup_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.1415926, 3.1415926],                 # FrustumMix 에 알맞게 rot_range 변경
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1]),
    dict(
        type='FrustumMix',                                 # Student network에 FrustumMix 적용
        H=64,
        W=512,
        fov_up=3.0,
        fov_down=-25.0,
        num_areas=[3, 4, 5, 6],
        pre_transform=pre_transform,
        prob=1.0),
    dict(
        type='InstanceCopy',
        instance_classes=[1, 2, 3, 4, 5, 6, 7, 11, 15, 17, 18],     # 클래스 설정후 변경해야하는 부분  labels_map 기준
        pre_transform=pre_transform,
        prob=1.0),
    dict(
        type='RangeInterpolation',
        H=64,
        W=2048,
        fov_up=3.0,
        fov_down=-25.0,
        ignore_index=19),
    dict(
        type='MultiBranch3D',                        # teacher-student network 구현
        branch_field=branch_field,
        sup=dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask']))
]

# pipeline used to augment unlabeled data,
# which will be sent to teacher model for predicting pseudo instances.
unsup_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1]),
    dict(
        type='MultiBranch3D',                        # teacher-student network 구현
        branch_field=branch_field,
        unsup=dict(
            type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask']))
]

segmentor = dict(
    type='FRNet',
    data_preprocessor=dict(type='FrustumRangePreprocessor',
                           H=64, W=512, fov_up=3.0, 
                           fov_down=-25.0, ignore_index=19
                           ),
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
        ignore_label=19,   # frnet 코드 참조하여 추가
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
        data_preprocessor=segmentor['data_preprocessor']   #FRNet 전처리 사용
    ),

    loss_mse=(dict(type='mmdet.MSELoss', loss_weight=250)),
    semi_train_cfg=dict(
        freeze_teacher=True, pseudo_thr=0.9, ignore_label=19,
        pitch_angles=[-25, 3], num_areas=[4, 5, 6, 7, 8],
        sup_weight=1, unsup_weight=1,
    ),
    semi_test_cfg=dict(extract_feat_on='teacher', predict_on='teacher'))

# quota
labeled_dataset = dict(
    type='SemanticKittiDataset',  
    data_root=data_root, pipeline=sup_pipeline, metainfo=metainfo,
    modality=input_modality, ignore_index=19, backend_args=backend_args,
    ann_file='semantickitti_infos_train.10.pkl'
)
unlabeled_dataset = dict(
    type='SemanticKittiDataset',
    data_root=data_root, pipeline=unsup_pipeline, metainfo=metainfo,
    modality=input_modality, ignore_index=19, backend_args=backend_args,
    ann_file='semantickitti_infos_train.10-unlabeled.pkl',
)
train_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(
        type='mmdet.MultiSourceSampler', batch_size=4, source_ratio=[1, 1],
    ),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset],
    )
)



"""
frnet 방식으로 수정
"""
# learning rate
lr = 0.01
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6),
    #clip_grad=dict(max_norm=10, norm_type=2),
)

param_scheduler = [
    dict(
        type='OneCycleLR',
        total_steps=50000,
        by_epoch=False,
        eta_max=lr,
        pct_start=0.2,
        div_factor=25.0,
        final_div_factor=100.0)
]

train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=50000, val_interval=1000)



# default hook
default_hooks = dict(checkpoint=dict(by_epoch=False, save_best='miou', rule='greater'))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='mmdet.MeanTeacherHook', momentum=0.01)]
