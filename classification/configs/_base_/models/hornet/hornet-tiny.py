# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(type="HorNet", arch="tiny", drop_path_rate=0.2),
    head=dict(
        type="LinearClsHead",
        num_classes=1000,
        in_channels=512,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, mode="original"),
        cal_acc=False,
    ),
    init_cfg=[
        dict(type="TruncNormal", layer="Linear", std=0.02, bias=0.0),
        dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
        dict(type="Constant", layer=["LayerScale"], val=1e-6),
    ],
    train_cfg=dict(
        augments=[
            dict(type="BatchMixup", alpha=0.8, num_classes=1000, prob=0.5),
            dict(type="BatchCutMix", alpha=1.0, num_classes=1000, prob=0.5),
        ]
    ),
)
