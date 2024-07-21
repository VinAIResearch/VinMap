# Model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="PoolFormer",
        arch="m36",
        drop_path_rate=0.1,
        init_cfg=[
            dict(type="TruncNormal", layer=["Conv2d", "Linear"], std=0.02, bias=0.0),
            dict(type="Constant", layer=["GroupNorm"], val=1.0, bias=0.0),
        ],
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=1000,
        in_channels=768,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
