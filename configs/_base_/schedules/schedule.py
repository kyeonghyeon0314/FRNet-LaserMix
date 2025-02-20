# optimizer
# This schedule is mainly used by models on indoor dataset,
# e.g., VoteNet on SUNRGBD and ScanNet

"""
iteration based training loop과 epoch based training loop 선택

현재 
iteration based training loop
"""


lr = 0.01  # max learning rate
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6),
    #clip_grad=dict(max_norm=10, norm_type=2),
)

# learning rate
"""
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 32],
        gamma=0.1)
]
"""
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



# training schedule for 3x
"""
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
"""
train_cfg = dict(type='IterBasedTrainLoop', max_iters=50000, val_interval=1000)
val_cfg = dict()
test_cfg = dict()



# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (4 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)