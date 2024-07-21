log_config = dict(interval=5, hooks=[dict(type="TextLoggerHook")])
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
opencv_num_threads = 0
mp_start_method = "fork"
model = dict(
    type="OCRMaskRCNN",
    pretrained="open-mmlab://resnext101_64x4d",
    backbone=dict(
        type="mmdet.ResNeXt",
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        groups=64,
        base_width=4,
        dcn=dict(type="DCNv2", deform_groups=4, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
    ),
    neck=dict(type="mmdet.FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator", scales=[4], ratios=[0.17, 0.44, 1.13, 2.9, 7.46], strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    roi_head=dict(
        type="StandardRoIHead",
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
        mask_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        mask_head=dict(
            type="FCNMaskHead",
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                gpu_assign_thr=50,
            ),
            sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(type="OHEMSampler", num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=200,
            nms_post=200,
            max_per_img=100,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type="nms", iou_threshold=0.4),
            max_per_img=50,  # adjusting these affects GPU usage
            mask_thr_binary=0.25,
        ),
    ),
)
optimizer = dict(type="SGD", lr=0.08, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[80, 128])
runner = dict(type="EpochBasedRunner", max_epochs=150)
checkpoint_config = dict(interval=1)
dataset_type = "IcdarDataset"
data_root = "data/icdar2015"
train = dict(
    type="IcdarDataset",
    ann_file="data/icdar2015/instances_training.json",
    img_prefix="data/icdar2015/imgs",
    pipeline=None,
)
test = dict(
    type="IcdarDataset", ann_file="data/icdar2015/instances_test.json", img_prefix="data/icdar2015/imgs", pipeline=None
)
train_list = [
    dict(
        type="IcdarDataset",
        ann_file="data/icdar2015/instances_training.json",
        img_prefix="data/icdar2015/imgs",
        pipeline=None,
    )
]
test_list = [
    dict(
        type="IcdarDataset",
        ann_file="data/icdar2015/instances_test.json",
        img_prefix="data/icdar2015/imgs",
        pipeline=None,
    )
]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile", color_type="color_ignore_orientation"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="ScaleAspectJitter",
        img_scale=None,
        keep_ratio=False,
        resize_type="indep_sample_in_range",
        scale_range=(640, 2560),
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type="RandomCropInstances", target_size=(640, 640), mask_type="union_all", instance_key="gt_masks"),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]
img_scale_ctw1500 = (1600, 1600)
test_pipeline_ctw1500 = [
    dict(type="LoadImageFromFile", color_type="color_ignore_orientation"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1600, 1600),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
img_scale_icdar2015 = (1920, 1920)
test_pipeline_icdar2015 = [
    dict(type="LoadImageFromFile", color_type="color_ignore_orientation"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1920, 1920),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type="UniformConcatDataset",
        datasets=[
            dict(
                type="IcdarDataset",
                ann_file="Dataset/json_converted/instances_training.json",
                img_prefix="Dataset/images",
                pipeline=None,
            )
        ],
        pipeline=[
            dict(type="LoadImageFromFile", color_type="color_ignore_orientation"),
            dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
            dict(
                type="ScaleAspectJitter",
                img_scale=None,
                keep_ratio=False,
                resize_type="indep_sample_in_range",
                scale_range=(640, 2560),
            ),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(type="Normalize", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type="RandomCropInstances", target_size=(640, 640), mask_type="union_all", instance_key="gt_masks"),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
        ],
    ),
    val=dict(
        type="UniformConcatDataset",
        datasets=[
            dict(
                type="IcdarDataset",
                ann_file="Dataset/json_converted/instances_validation.json",
                img_prefix="Dataset/images",
                pipeline=None,
            )
        ],
        pipeline=[
            dict(type="LoadImageFromFile", color_type="color_ignore_orientation"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(1920, 1920),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(type="Normalize", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
    test=dict(
        type="UniformConcatDataset",
        datasets=[
            dict(
                type="IcdarDataset",
                ann_file="Dataset/json_converted/instances_testing.json",
                img_prefix="Dataset/images",
            ),
        ],
        pipeline=[
            dict(type="LoadImageFromFile", color_type="color_ignore_orientation"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(1920, 1920),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(type="Normalize", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
)
evaluation = dict(interval=160, metric="hmean-iou", save_best="0_hmean-iou_hmean", rule="greater")
work_dir = "work_dirs/maskrcnn_resnext101_DCN_160e_icdar"
gpu_ids = [0]
