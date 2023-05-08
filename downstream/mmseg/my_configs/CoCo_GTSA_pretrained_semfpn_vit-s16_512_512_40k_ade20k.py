
_base_ = [
    "/workspace/mmsegmentation/configs/_base_/datasets/ade20k.py",
    "/workspace/mmsegmentation/configs/_base_/default_runtime.py",
    "/workspace/mmsegmentation/configs/_base_/schedules/schedule_40k.py",
]
crop_size = (512, 512)

norm_cfg = dict(type="SyncBN", requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VisionTransformer',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=384,
        num_layers=12,
        num_heads=6,
        mlp_ratio=4,
        out_indices=(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        interpolate_mode='bicubic',
        #pretrained="/nasdata3/9kth/GTE_SSL/my_pretrain/GE_smooth_SSL/CoCo_GE_smooth_SSL_99ep_mm_style.pth",
        pretrained="/workspace/final/GTSA_exps/my_pretrain/GTSA/CoCo_GTSA_99ep_mm_style.pth",

        #init_cfg='normals',
    ),
    neck=dict(
        type='FPN',
        in_channels=[384, 384, 384, 384],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))  
)
    
#test_cfg=dict(mode='whole', crop_size=crop_size, stride=(341, 341)),
test_cfg=dict(mode='whole')
init_cfg=dict(type="Pretrained", checkpoint="")


optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
