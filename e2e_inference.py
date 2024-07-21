import argparse
import json

# from google.colab.patches import cv2_imshow
import math
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mmocr.utils.ocr import MMOCR
from PIL import Image, ImageDraw, ImageFont
from tools.minimum_hull import minimum_bounding_rectangle
from tqdm import tqdm
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Convert JSON annotations to ICDAR format")
    parser.add_argument("-config", "--config", help="Testing Configuration")
    parser.add_argument("-det_weights", "--det_weights", help="Detection Weight")
    parser.add_argument("-rec_weights", "--rec_weights", help="Recognition Weight")
    parser.add_argument("-input_images", "--input_images", help="Input images path")
    parser.add_argument("-output_destination", "--output_destination", help="Output path")
    args = parser.parse_args()
    return args


def excecute(folder_path, image_folder_path):
    print("Excecuting OCR")
    os.mkdir(folder_path + "/" + "predicted")
    for f in tqdm(os.listdir(image_folder_path)):
        file_type = f.split(".")[1]
        file_name = f.split(".")[0]

        image = cv2.imread(os.path.join(image_folder_path, f))
        ff = None
        try:
            ff = open(os.path.join(folder_path, "out_" + file_name + ".json"), "r")
        except:
            continue
        all_data = json.load(ff)

        boundary_results = all_data["boundary_result"]
        results = []
        ii = Image.fromarray(np.array(image))
        draw = ImageDraw.Draw(ii)
        for boundary_result in boundary_results:
            np_image = np.array(image)
            info = []
            if boundary_result[-1] <= 0.25:
                continue
            points = []
            for i in range(0, len(boundary_result) - 2, 2):
                points.append(tuple([int(boundary_result[i]), int(boundary_result[i + 1])]))
            points = np.array(points)
            try:
                four_points = minimum_bounding_rectangle(points)
            except:
                continue
            four_points = np.array(four_points).astype(int)

            rect = cv2.minAreaRect(four_points)
            box = cv2.boxPoints(rect)
            oriented_rec = np.int0(box)

            x_tl, y_tl = min(oriented_rec[:, 0]), min(oriented_rec[:, 1])
            x_br, y_br = max(oriented_rec[:, 0]), max(oriented_rec[:, 1])
            if x_tl < 0 or y_tl < 0 or x_br >= np_image.shape[1] or y_br >= np_image.shape[0]:
                np_image = cv2.copyMakeBorder(np_image, 500, 500, 500, 500, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                np_image = np_image[y_tl + 500 : y_br + 500, x_tl + 500 : x_br + 500]
            else:
                np_image = np_image[y_tl:y_br, x_tl:x_br]

            try:
                s = detector.predict(Image.fromarray(np_image))
            except:
                continue

            font_path = os.path.join(cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf")
            font = ImageFont.truetype(font_path, size=16)
            draw.text((x_tl, y_tl), str(s), fill="red", font=font)
            draw.rectangle([x_tl, y_tl, x_br, y_br], outline="blue")

            clockwise = np.flip(oriented_rec, axis=0)
            for p in clockwise:
                info.append(str(p[0]))
                info.append(str(p[1]))

            info.append(str(s))
            results.append(",".join(info))
        # ii.save(folder_path + '/drive/' + file_name + '_txt_.jpg')

        file_submit_name = os.path.join(folder_path + "/" + "predicted", file_name + ".txt")
        with open(file_submit_name, "w") as file_submit:
            for line_string in results:
                file_submit.write(line_string)
                file_submit.write("\n")


if __name__ == "__main__":
    args = parse_args()
    hack = 1000  ### not hack == 1
    while hack >= 1:
        config = args.config
        detw = args.det_weights
        recw = args.rec_weights
        inp = args.input_images
        out = args.output_destination
        ## Detection
        # Load models into memory
        ocr = MMOCR(det="MaskRCNN_IC15", det_config=config, recog=None, det_ckpt=detw)
        print("Completed Loading Det Model")
        # Inference
        results = ocr.readtext(inp, output=out, export=out)
        print("Completed Testing Detection")
        ## Recognition
        config = Cfg.load_config_from_name("vgg_transformer")
        config["weights"] = recw
        config["cnn"]["pretrained"] = False
        config["device"] = "cuda:0"
        detector = Predictor(config)
        print("Completed Loading Rec Model")
        # excecute(out,inp)
        print("Finised")

        hack -= 1
