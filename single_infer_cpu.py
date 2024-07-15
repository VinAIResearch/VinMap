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
sys.path.append('/home/phucnda/applied/E2E_WordArt_DetRec/')

# segmentation
from tqdm import tqdm
from mmocr.utils.ocr import MMOCR
# recognition
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from tools.minimum_hull import minimum_bounding_rectangle
from PIL import Image, ImageDraw, ImageFont
# classification
import mmcv
from classification.mmcls.apis import inference_model, init_model, show_result_pyplot
from mmcv import Config
from classification.mmcls.utils import auto_select_device
from classification.mmcls.apis.inference import data_infer

# vocab_filtering
from eval import check
from eval import mapper

import shutil
from unidecode import unidecode

def parse_args():
    parser = argparse.ArgumentParser(
        description='Release-21.01.16'
    )
    
    parser.add_argument('-configCls','--configCls', default = 'classification/configs/efficientnet/efficientnet-b4_8xb32_in1k.py', help='Map Classification Configuration')
    parser.add_argument('-configSegment','--configSegment', default = 'configuration/maskrcnn_resnext101_DCN_160e_icdar_cpu.py', help='Text Segmentation Configuration')

    parser.add_argument('-cls_weights','--cls_weights', default = '../weight/effnetb4_vnmap.pth', help='Map Classification Weight')
    parser.add_argument('-det_weights','--det_weights', default ='../weight/resnext101_DCN_160e_epoch_150.pth', help='Text Segmentation Weight')
    parser.add_argument('-rec_weights','--rec_weights', default = '../weight/transformerocr_btc.pth', help='Text Recognition Weight')

    parser.add_argument('--root','--root', default = 'None', help='Root Workdirectory')
    parser.add_argument('-input_images','--input_images',default = 'None', help='Input images path')
    parser.add_argument('-output_destination','--output_destination',default = 'None', help='Output path')

    parser.add_argument('-single_infer','--single_infer',default = 'True', help='Type of single infering')    
    parser.add_argument('-single_infer_image','--single_infer_image',default = 'None', help='Input path of single image')    
    parser.add_argument('-single_infer_path','--single_infer_path',default = 'None', help='Output path of single infer')    
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
        self.configSegment = args.configSegment
        
        self.cls_weights = args.cls_weights
        self.det_weights = args.det_weights
        self.rec_weights = args.rec_weights
        
        self.root = args.root
        self.input_images = args.input_images
        self.output_destination = args.output_destination

        self.device = 'cpu'
        # Setting up configuration

        self.SegmentationModel = MMOCR(det='MaskRCNN_IC15', det_config=self.configSegment, recog=None, det_ckpt=self.det_weights, device = self.device)
        print('Completed Loading Det Model')

    def e2e_infer(self, filename, folder_path):
        '''
        e2e Inferences
            filename: image file path
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
        
        seg_res = self.SegmentationModel.readtext(filename, output = folder_path+'/'+'fig', export = folder_path+'/'+'fig')
        self.SegmentationModel = None
        torch.cuda.empty_cache()
        
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
        self.ClsModel = init_model(cfg, self.cls_weights, device=device)
        self.ClsModel.CLASSES = ['notvietnam', 'vietnam']
        self.ClsModel.cfg = cfg
        print('Completed Loading Cls Model')        
        self.OCRModel = Predictor(config_rec)
        print('Completed Loading Rec Model')

        results = 0
        image_folder_path = folder_path+'/'+'fig'
        for f in (os.listdir(image_folder_path)):
            # Read image
            try:
                image = cv2.imread(filename)
            except:
                continue

            file_type = f.split(".")[-1]
            if file_type != 'json':
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
                continue
            ff = None
            try:
                ff = open(os.path.join(image_folder_path, f), 'r')
            except:
                results = 0
                continue
            all_data = json.load(ff)

            boundary_results = all_data['boundary_result']
            result_str = []
            ii = Image.fromarray(np.array(image))

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
            print('This is VN map. It does not contain (Truong Sa and Hoang Sa).. ALERT')
        elif (results == 1):
            print('This is not VN map.. SKIPPED')
        elif (results == 2):
            print('This is VN map, It contains (Truong Sa or Hoang Sa).. OK')

        

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
        device = 'cpu'
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
            device = 'cpu'
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


if __name__ == "__main__":
    args = parse_args()
    wrapper = ModelWrapper(args)
    if args.single_infer == 'False':
        wrapper.process_image()
    else:
        wrapper.e2e_infer(args.single_infer_image, args.single_infer_path)