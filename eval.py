import argparse
import os
import json
import time
import cv2
# from google.colab.patches import cv2_imshow
import math
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
# from mmocr.utils.ocr import MMOCR
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from tools.minimum_hull import minimum_bounding_rectangle
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

mapper = [
    ['hoangsa'],
    ['truongsa'],
    ['hoangsa', 'truongsa'],
    ['hoang', 'sa'],
    ['truong', 'sa'],
    ['hoang', 'sa', 'truong', 'sa'],
    ['hoàng', 'sa', 'trường', 'sa'],
    ['hoang', 'sa', 'trương', 'sa'],
    ['hoàng', 'sa', 'trương', 'sa'],
    ['hoàng', 'sa'],
    ['trường', 'sa'],
    ['trương', 'sa'],
    ['quan', 'dao', 'truong'],
    ['quan', 'dao', 'hoang'],
    ['paracel'],
    ['spratly']
]

def check(word_bag):
    global mapper
    for cond in mapper:
        count_dict = Counter(cond)
        for word in word_bag:
            if word in count_dict:
                count_dict[word] -= 1
        f = 1
        for element, count in count_dict.items():
            if count > 0:
                f = 0
        if f == 1:
            return 1
    return 0

eval = False
if eval == True:
    ### POSITIVE GT
    path = '../Dataset/Vietnam_map/prediction/positive/vietnamese/predicted'
    files = os.listdir(path)
    pos = 0
    neg = 0
    for file in files:
        word_bag = []
        with open(path + '/' + file, 'r') as file:
            lines = file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
            word_bag.append(lines[i].split(',')[-1].lower())
        if check(word_bag) == 1:
            neg += 1
        else:
            pos += 1

    path = '../Dataset/Vietnam_map/prediction/positive/english/predicted'
    files = os.listdir(path)
    for file in files:
        word_bag = []
        with open(path + '/' + file, 'r') as file:
            lines = file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
            word_bag.append(lines[i].split(',')[-1].lower())
        if check(word_bag) == 1:
            neg += 1
        else:
            pos += 1
    TP = pos
    FN = neg

    ### NEGATIVE GT
    path = '../Dataset/Vietnam_map/prediction/negative/vietnamese/predicted'
    files = os.listdir(path)
    pos = 0
    neg = 0
    for file in files:
        word_bag = []
        with open(path + '/' + file, 'r') as file:
            lines = file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
            word_bag.append(lines[i].split(',')[-1].lower())
        if check(word_bag) == 1:
            neg += 1
        else:
            pos += 1

    path = '../Dataset/Vietnam_map/prediction/negative/english/predicted'
    files = os.listdir(path)
    for file in files:
        word_bag = []
        with open(path + '/' + file, 'r') as file:
            lines = file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
            word_bag.append(lines[i].split(',')[-1].lower())
        if check(word_bag) == 1:
            neg += 1
        else:
            pos += 1
    FP = pos
    TN = neg

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2*(Precision*Recall)/(Precision+Recall)

    print('Precision:', Precision)
    print('Recall:', Recall)
    print('F1:', F1)
