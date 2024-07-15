import mmcv
from mmcls.apis import inference_model, init_model, show_result_pyplot

# Specify the path of the config file and checkpoint file.
config_file = 'configs/efficientnet/efficientnet-b4_8xb32_in1k.py'
checkpoint_file = 'weights/efficientnet-b4_3rdparty_8xb32_in1k_20220119-81fd4077.pth'

# Specify the device, if you cannot use GPU, you can also use CPU 
# by specifying `device='cpu'`.
device = 'cuda:0'
# device = 'cpu'

# Build the model according to the config file and load the checkpoint.
model = init_model(config_file, checkpoint_file, device=device)

dataset = '../../MapData'
data_root = '../../MapData'

# Load the base config file
from mmcv import Config
from mmcls.utils import auto_select_device

cfg = Config.fromfile(config_file)
cfg.device = auto_select_device()

# Modify the number of classes in the head.
cfg.model.head.num_classes = 2
cfg.model.head.topk = (1, )

# Load the pre-trained model's checkpoint.
cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone')

# Specify sample size and number of workers.
cfg.data.samples_per_gpu = 4
cfg.data.workers_per_gpu = 2

# Specify the path and meta files of training dataset
cfg.data.train.data_prefix = dataset
cfg.data.train.classes = data_root + '/classes.txt'

# Specify the path and meta files of validation dataset
cfg.data.val.data_prefix = dataset
cfg.data.val.ann_file = data_root + '/label.txt'
cfg.data.val.classes = data_root + '/classes.txt'

# Specify the path and meta files of test dataset
cfg.data.test.data_prefix = dataset
cfg.data.test.ann_file = data_root + '/label.txt'
cfg.data.test.classes = data_root + '/classes.txt'

cfg.runner = dict(type='EpochBasedRunner', max_epochs=100)

# Specify the work directory
cfg.work_dir = './work_dirs/map_dataset/effnetb4'

# Output logs for every 10 iterations
cfg.log_config.interval = 10

# Set the random seed and enable the deterministic option of cuDNN
# to keep the results' reproducible.
from mmcls.apis import set_random_seed
cfg.seed = 0
set_random_seed(0, deterministic=True)

cfg.gpu_ids = range(1)


import time
import mmcv
import os.path as osp

from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.apis import train_model

# Create the work directory
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# Build the classifier
model = build_classifier(cfg.model)
model.init_weights()
# Build the dataset
datasets = [build_dataset(cfg.data.train)]
# Add `CLASSES` attributes to help visualization
model.CLASSES = datasets[0].CLASSES
# Start fine-tuning
train_model(
    model,
    datasets,
    cfg,
    distributed=False,
    validate=False,
    timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
    meta=dict())