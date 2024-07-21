# Import Library
import matplotlib.pyplot as plt
from PIL import Image
from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


# Training

config = Cfg.load_config_from_name("vgg_transformer")
dataset_params = {
    "name": "hw",
    "data_root": "data/WordArt/all_image",
    "train_annotation": "train_label.txt",
    "valid_annotation": "test_label.txt",
}

params = {
    "batch_size": 64,
    "print_every": 200,
    "valid_every": 15 * 200,
    "iters": 20000,
    "checkpoint": "weights/transformerocr.pth",
    "export": "weights/transformerocr_wordart64.pth",
    "metrics": 10000,
}

worker = {"num_workers": 3}

config["trainer"].update(params)
config["dataset"].update(dataset_params)
config["dataloader"].update(worker)
config["device"] = "cuda:4"

trainer = Trainer(config, pretrained=True)
trainer.config.save("config.yml")
trainer.train()
