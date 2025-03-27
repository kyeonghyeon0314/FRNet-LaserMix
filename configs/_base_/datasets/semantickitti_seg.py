# For SemanticKitti we usually do 19-class segmentation.
# For labels_map we follow the uniform format of MMDetection & MMSegmentation
# i.e. we consider the unlabeled class as the last one, which is different
# from the original implementation of some methods e.g. Cylinder3D.
dataset_type = 'SemanticKittiDataset'
data_root = 'data/semantickitti/'


# 수정의 필요가 있어 보이지만 일단 보류
class_names = [
    'car', 'road', 'sidewalk', 'other-vehicle'
]

"""
19 : unlabeled  --> 4
0 : car
8 : road  --> 1
10 : sidewalk --> 2
4 : other-vehicle --> 3

labels_map 위 클래스로 수정완료

""" 
# 라벨 데이터 remap

labels_map ={
    0 : 4,   # unlabeled
    10 : 0,  # car
    40 : 1,  # road
    48 : 2,  # sidewalk
    20 : 3,  # other-vehicle

}

"""
현재 반지도학습을 위해 lasermix 정보를 가지고 왔음 cylinder3d를 frnet으로 변경 적용
Teacher-student Network 기반의 완전 지도 학습 , 선행학습된 Teacher model을 통한 지도 학습 적용
"""

metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=259  # max_label= 259
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
        type='InstanceCopy',                         # 데이터수가 적은 개별 객체를 위한 데이터 증강기법
        instance_classes=[3],                        # 개별 객체의 단위 복사 클래스 other-vehicle만 사용
        pre_transform=pre_transform,
        prob=1.0),
    dict(
        type='RangeInterpolation',
        H=64,
        W=2048,
        fov_up=3.0,
        fov_down=-25.0,
        ignore_index=4),
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
        rot_range=[-3.1415926, 3.1415926],  # teacher-network 에서는 회전범위 축소
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1]),
    dict(
        type='MultiBranch3D',                        # teacher-student network 구현
        branch_field=branch_field,
        unsup=dict(
            type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask']))
]

test_pipeline = [
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
        type='RangeInterpolation',
        H=64,
        W=2048,
        fov_up=3.0,
        fov_down=-25.0,
        ignore_index=4),
    dict(type='Pack3DDetInputs', keys=['points'], meta_keys=['num_points'])
]

# for prediction.py
test_pipeline_2 = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='RangeInterpolation',
        H=64,
        W=2048,
        fov_up=3.0,
        fov_down=-25.0,
        ignore_index=4),
    dict(type='Pack3DDetInputs', keys=['points'], meta_keys=['num_points'])
]

tta_pipeline = [
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
        type='RangeInterpolation',
        H=64,
        W=2048,
        fov_up=3.0,
        fov_down=-25.0,
        ignore_index=4),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=1.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=1.)
        ],
                    [
                        dict(
                            type='GlobalRotScaleTrans',
                            rot_range=[-3.1415926, 3.1415926],
                            scale_ratio_range=[0.95, 1.05],
                            translation_std=[0.1, 0.1, 0.1])
                    ], 
                    [
                        dict(type='Pack3DDetInputs', keys=['points'],meta_keys=['num_points'])]])
]

labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='semantickitti_infos_train.pkl',
    pipeline=sup_pipeline,
    metainfo=metainfo,
    modality=input_modality,
    ignore_index=4,
    backend_args=backend_args)

# unlabeled_dataset을 활용하지 않으므로 처리해야하는 부분분
labeled_dataset_teacher = dict(
    type=dataset_type,
    data_root=data_root, 
    pipeline=unsup_pipeline, 
    metainfo=metainfo,
    modality=input_modality, 
    ignore_index=4, 
    backend_args=backend_args,
    ann_file='semantickitti_infos_train.pkl',
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='mmdet.MultiSourceSampler', batch_size=2, source_ratio=[1, 1]),
    dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, labeled_dataset_teacher])    # labeled data만 적용 할지 선택
    #dataset=labeled_dataset # labeled_dataset
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='semantickitti_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=4,
        test_mode=True,
        backend_args=backend_args))

# for prediction.py
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='semantickitti_infos_trainval.pkl',
        pipeline=test_pipeline_2,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=4,
        test_mode=True,
        backend_args=backend_args))

val_evaluator = dict(type='SegMetric')
#test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

tta_model = dict(type='Seg3DTTAModel')




