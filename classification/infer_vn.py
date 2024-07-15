import mmcv
from mmcls.apis import inference_model, init_model, show_result_pyplot

# Specify the path of the config file and checkpoint file.
config_file = 'configs/efficientnet/efficientnet-b4_8xb32_in1k.py'
checkpoint_file = 'work_dirs/vietnam_dataset/effnetb4/latest.pth'


# Load the base config file
from mmcv import Config
from mmcls.utils import auto_select_device
from mmcls.models import build_classifier
cfg = Config.fromfile(config_file)
cfg.device = auto_select_device()
# Modify the number of classes in the head.
cfg.model.head.num_classes = 2
cfg.model.head.topk = (1, )
# Load the pre-trained model's checkpoint.
cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone')
cfg.model.head.num_classes = 2
cfg.model.head.topk = (1, )

# Build the classifier
device = 'cuda:0'
# Build the model according to the config file and load the checkpoint.
model = init_model(cfg, checkpoint_file, device=device)
model.CLASSES = ['notvietnam', 'vietnam']

model.cfg = cfg
img = '../../Dataset/MapT12_classification/notvietnam/00001.jpg'
img_array =  mmcv.imread(img)
result = inference_model(model, img_array)

result