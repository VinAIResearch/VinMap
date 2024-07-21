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
    "data_root": "data/images",
    "train_annotation": "anno.txt",
    "valid_annotation": "anno.txt",
}

params = {
    "batch_size": 128,
    "print_every": 200,
    "valid_every": 15 * 200,
    "iters": 100000,
    "checkpoint": "weights/transformerocr_btcc.pth",
    "export": "weights/transformerocr_combine.pth",
    "metrics": 10000,
}

worker = {"num_workers": 3}

config["trainer"].update(params)
config["dataset"].update(dataset_params)
config["dataloader"].update(worker)
config["device"] = "cuda:5"

trainer = Trainer(config, pretrained=True)
trainer.config.save("config.yml")
trainer.train()
