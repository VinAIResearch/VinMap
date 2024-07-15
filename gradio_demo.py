import argparse
import os
import json
import time
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
sys.path.append('/root/VinMap')

# segmentation
from tqdm import tqdm
# recognition
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from tools.minimum_hull import minimum_bounding_rectangle
from PIL import Image, ImageDraw, ImageFont
# classification
from classification.mmcls.apis import inference_model, init_model, show_result_pyplot
from mmcv import Config
from classification.mmcls.utils import auto_select_device
from classification.mmcls.apis.inference import data_infer

# vocab_filtering
from eval import check
from eval import mapper

import shutil
from unidecode import unidecode

#
from PaddleOCR.tools.infer.predict_det_our import readtext
import ast
import json

import gradio as gr

def parse_args():
    parser = argparse.ArgumentParser(
        description='Release-24.03.12'
    )
    
    parser.add_argument('-configCls','--configCls', default = 'classification/configs/efficientnet/efficientnet-b4_8xb32_in1k.py', help='Map Classification Configuration')

    parser.add_argument('-map_weights','--map_weights', default = '../weight/effnetb4_imagemap.pth', help='Map Classification Weight')
    parser.add_argument('-cls_weights','--cls_weights', default = '../weight/effnetb4_vnmap.pth', help='VN Map Classification Weight')
    parser.add_argument('-rec_weights','--rec_weights', default = '../weight/transformerocr_btc.pth', help='Text Recognition Weight')

    parser.add_argument('-mass_dir','--mass_dir',default = 'None', help='Input path of mass image directory')    
    parser.add_argument('-output_json','--output_json',default = 'None', help='Output path')
    parser.add_argument('-infer_path','--infer_path',default = '../temp', help='Output path of image infer')    

    args = parser.parse_args()
    return args

def excecute(root, folder_path,image_folder_path):
  print('Excecuting OCR')
  os.makedirs(folder_path+'/'+'predicted')
  for f in tqdm(os.listdir(image_folder_path)):
    file_type = f.split(".")[1]
    file_name = f.split(".")[0]

    image = cv2.imread(os.path.join(image_folder_path, f))
    ff = None
    try:
        ff = open(os.path.join(folder_path, "out_" + file_name + '.json'), 'r')
    except:
        continue
    all_data = json.load(ff)

    boundary_results = all_data['boundary_result']
    results = []
    ii = Image.fromarray(np.array(image))
    draw = ImageDraw.Draw(ii)
    for boundary_result in boundary_results:
        np_image = np.array(image)
        info = []
        if (boundary_result[-1] < 0.1):
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
        if x_tl<0 or y_tl<0 or x_br>=np_image.shape[1] or y_br>=np_image.shape[0]:
            np_image = cv2.copyMakeBorder(np_image, 500, 500, 500, 500, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            np_image = np_image[ y_tl + 500:y_br + 500, x_tl + 500:x_br + 500 ]
        else:
            np_image = np_image[ y_tl:y_br, x_tl:x_br   ]
    
        try:
            s = detector.predict(Image.fromarray(np_image))
        except:
            continue

        font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
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
    
    file_submit_name = os.path.join(folder_path+'/'+'predicted', file_name + ".txt") 
    with open(file_submit_name, "w") as file_submit:
        for line_string in results:
            file_submit.write(line_string)
            file_submit.write('\n')

class ModelWrapper():
    def __init__(self, args):
        self.configCls = args.configCls
        
        self.map_weights = args.map_weights
        self.cls_weights = args.cls_weights
        self.rec_weights = args.rec_weights
        
        self.input_images = args.mass_dir
        self.mass_dir = args.mass_dir
        self.output_json = args.output_json

        self.device = 'cuda:0'
        # Setting up configuration

    def e2e_infer(self, filenames, folder_path):
        '''
        e2e Inferences
            filename: image files path
            folder_path: prediction archive
        Output: Resulting
            Postitive: 
                (VN map) and 
                do not contain (Truong Sa and Hoang Sa)
            Negative:
                (not VN map) or
                (VN map and contain (Truong Sa or Hoang Sa))
            
            Results:
                0: Positive
                1: Negative1
                2: Negative2
        '''
        if os.path.exists(folder_path+'/'+'predicted') == False:
            os.makedirs(folder_path+'/'+'predicted')
        else:
            shutil.rmtree(folder_path+'/'+'predicted')
            os.makedirs(folder_path+'/'+'predicted')

        if os.path.exists(folder_path+'/'+'fig') == False:
            os.makedirs(folder_path+'/'+'fig')
        else:
            shutil.rmtree(folder_path+'/'+'fig')
            os.makedirs(folder_path+'/'+'fig')
        files = os.listdir(filenames)
        
        diction = {} # result dictionary


            ## Recognition
        device = self.device
        config_rec = Cfg.load_config_from_name('vgg_transformer')
        config_rec['weights'] = self.rec_weights
        config_rec['cnn']['pretrained']=False
        config_rec['device'] = device
            ## Classification
        cfg = Config.fromfile(self.configCls)
        cfg.device = device
        cfg.model.head.num_classes = 2
        cfg.model.head.topk = (1, )
        cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint=self.cls_weights, prefix='backbone')
        cfg.model.head.num_classes = 2
        cfg.model.head.topk = (1, )

        # Model load weights

        self.MapModel = init_model(cfg, self.map_weights, device=device)
        self.MapModel.CLASSES = ['map', 'notmap']
        self.MapModel.cfg = cfg

        self.ClsModel = init_model(cfg, self.cls_weights, device=device)
        self.ClsModel.CLASSES = ['notvietnam', 'vietnam']
        self.ClsModel.cfg = cfg
        print('Completed Loading Cls Model')        
        self.OCRModel = Predictor(config_rec)
        print('Completed Loading Rec Model')


        for filenamei in tqdm(files):
            filename = os.path.join(filenames, filenamei)
            readtext(folder_path+'/'+'fig', filename)
            torch.cuda.empty_cache()

            results = 0
            image_folder_path = folder_path+'/'+'fig'
            for f in (os.listdir(image_folder_path)):
                # Read image
                try:
                    image = cv2.imread(filename)
                except:
                    continue

                file_type = f.split(".")[-1]

                ## Image - Map CLASSIFICATION

                data = data_infer(self.MapModel.cfg, filename, device)
                # forward the model
                with torch.no_grad():
                    scores = self.MapModel(return_loss=False, **data)
                    pred_score = np.max(scores, axis=1)[0]
                    pred_label = np.argmax(scores, axis=1)[0]
                    result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
                result['pred_class'] = self.MapModel.CLASSES[result['pred_label']]
                if result ['pred_label'] == 1: # Not VietNam
                    results = 1
                    diction[filename] = 'This is not a map.. SKIPPED'
                    continue
                
                ## CLASSIFICATION
                data = data_infer(self.ClsModel.cfg, filename, device)
                # forward the model
                with torch.no_grad():
                    scores = self.ClsModel(return_loss=False, **data)
                    pred_score = np.max(scores, axis=1)[0]
                    pred_label = np.argmax(scores, axis=1)[0]
                    result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
                result['pred_class'] = self.ClsModel.CLASSES[result['pred_label']]
                if result ['pred_label'] == 0: # Not VietNam
                    results = 1
                    diction[filename] = 'This is not VN map.. SKIPPED'
                    continue
                ff = None
                text_file = os.path.join(image_folder_path, 'det_results.txt')
                f = open(text_file, "r")
                tmp = f.read()
                boundary_results = ast.literal_eval(tmp.split('\t')[1].strip())

                result_str = []
                ii = Image.fromarray(np.array(image))

                for boundary_result in boundary_results:
                    np_image = np.array(image)
                    info = []
                    points = []
                    for i in range(0, len(boundary_result), 1):
                        points.append(tuple([int(boundary_result[i][0]), int(boundary_result[i][1])])) 
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
                    if x_tl<0 or y_tl<0 or x_br>=np_image.shape[1] or y_br>=np_image.shape[0]:
                        np_image = cv2.copyMakeBorder(np_image, 500, 500, 500, 500, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        np_image = np_image[ y_tl + 500:y_br + 500, x_tl + 500:x_br + 500 ]
                    else:
                        np_image = np_image[ y_tl:y_br, x_tl:x_br]
                
                    try:
                        s = self.OCRModel.predict(Image.fromarray(np_image))
                    except:
                        continue

                    clockwise = np.flip(oriented_rec, axis=0)
                    for p in clockwise:
                        info.append(str(p[0]))
                        info.append(str(p[1]))
                    
                    info.append(str(s))
                    result_str.append(unidecode(info[-1].lower()))
                if check(result_str) == 1:
                    results = 2
                else:
                    results = 0

            # Postitive: 
            #     (VN map) and 
            #     do not contain (Truong Sa and Hoang Sa)
            # Negative:
            #     (not VN map) or
            #     (VN map and contain (Truong Sa or Hoang Sa))
            if results == 0:
                diction[filename] = 'This is VN map. It does not contain (Truong Sa and Hoang Sa).. ALERT' 
            elif (results == 1):
                diction[filename] = 'This is not VN map.. SKIPPED'
            elif (results == 2):
                diction[filename] = 'This is VN map, It contains (Truong Sa or Hoang Sa).. OK'
        
        if results == 1:
            out = image
        else:
            out = np.asarray(Image.open('../temp/fig/det_res_test.png'))
        return diction[filename], out
        # Serializing json
        json_object = json.dumps(diction, indent=4)
        # Writing to sample.json
        with open(self.output_json, "w") as outfile:
            outfile.write(json_object)

    def process_image(self):
        '''
        Mass Inference
        Outputing logging files -> final results from eval.py
        '''
        folder_path = self.output_destination
        image_folder_path = self.input_images
        if os.path.exists(folder_path+'/'+'predicted') == False:
            os.makedirs(folder_path+'/'+'predicted')
        seg_res = self.SegmentationModel.readtext(self.input_images, output = self.output_destination, export = self.output_destination)
        self.SegmentationModel = None
        torch.cuda.empty_cache()
            ## Recognition
        device = 'cuda:0'
        config_rec = Cfg.load_config_from_name('vgg_transformer')
        config_rec['weights'] = self.rec_weights
        config_rec['cnn']['pretrained']=False
        config_rec['device'] = device
            ## Classification
        cfg = Config.fromfile(self.configCls)
        cfg.device = auto_select_device()
        cfg.model.head.num_classes = 2
        cfg.model.head.topk = (1, )
        cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint=self.cls_weights, prefix='backbone')
        cfg.model.head.num_classes = 2
        cfg.model.head.topk = (1, )

        # Model load weights
        self.ClsModel = init_model(cfg, self.cls_weights, device=device)
        self.ClsModel.CLASSES = ['notvietnam', 'vietnam']
        self.ClsModel.cfg = cfg
        print('Completed Loading Cls Model')        
        self.OCRModel = Predictor(config_rec)
        print('Completed Loading Rec Model')

        for f in tqdm(os.listdir(image_folder_path)):
            file_type = f.split(".")[-1]
            file_name = '.'.join(f.split(".")[0:-1])
            ## CLASSIFICATION
            device = 'cuda:0'
            data = data_infer(self.ClsModel.cfg, os.path.join(image_folder_path, f), device)
            # forward the model
            with torch.no_grad():
                scores = self.ClsModel(return_loss=False, **data)
                pred_score = np.max(scores, axis=1)[0]
                pred_label = np.argmax(scores, axis=1)[0]
                result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
            result['pred_class'] = self.ClsModel.CLASSES[result['pred_label']]
            if result ['pred_label'] == 0: # Not VietNam
                file_submit_name = os.path.join(folder_path+'/'+'predicted', file_name + ".txt") 
                with open(file_submit_name, "w") as file_submit:
                    file_submit.write('notvnmap')
                    file_submit.write('\n')
                continue
            image = cv2.imread(os.path.join(image_folder_path, f))
            ff = None
            try:
                ff = open(os.path.join(folder_path, "out_" + file_name + '.json'), 'r')
            except:
                file_submit_name = os.path.join(folder_path+'/'+'predicted', file_name + ".txt") 
                with open(file_submit_name, "a") as file_submit:
                    file_submit.write('vnmap')
                    file_submit.write('\n')
                continue
            all_data = json.load(ff)

            boundary_results = all_data['boundary_result']
            results = []
            ii = Image.fromarray(np.array(image))
            # draw = ImageDraw.Draw(ii)
            for boundary_result in boundary_results:
                np_image = np.array(image)
                info = []
                if (boundary_result[-1] < 0.1):
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
                if x_tl<0 or y_tl<0 or x_br>=np_image.shape[1] or y_br>=np_image.shape[0]:
                    np_image = cv2.copyMakeBorder(np_image, 500, 500, 500, 500, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    np_image = np_image[ y_tl + 500:y_br + 500, x_tl + 500:x_br + 500 ]
                else:
                    np_image = np_image[ y_tl:y_br, x_tl:x_br]
            
                try:
                    s = self.OCRModel.predict(Image.fromarray(np_image))
                except:
                    continue

                # font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
                # font = ImageFont.truetype(font_path, size=16)
                # draw.text((x_tl, y_tl), str(s), fill="red", font=font)
                # draw.rectangle([x_tl, y_tl, x_br, y_br], outline="blue")

                clockwise = np.flip(oriented_rec, axis=0)
                for p in clockwise:
                    info.append(str(p[0]))
                    info.append(str(p[1]))
                
                info.append(str(s))
                results.append(",".join(info))
            file_submit_name = os.path.join(folder_path+'/'+'predicted', file_name + ".txt") 
            with open(file_submit_name, "w") as file_submit:
                for line_string in results:
                    file_submit.write(line_string)
                    file_submit.write('\n')

from PIL import Image
import shutil



def greet(image):
    args = parse_args()
    wrapper = ModelWrapper(args)
    os.makedirs(args.mass_dir, exist_ok  = True)
    Image.fromarray(image).save(os.path.join(args.mass_dir,'test.png'))
    text, im = wrapper.e2e_infer(args.mass_dir, args.infer_path)
    shutil.rmtree(args.mass_dir)
    return text, im

if __name__ == "__main__":

    demo = gr.Interface(
        fn=greet,
        inputs=["image"],
        outputs=["text", "image"],
        share=True
    )

    demo.launch()

