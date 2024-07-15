import cv2
from PIL import Image
import os

name = os.listdir('../Dataset/Vietnam_map/negative/english')
for n in name:
    print(n)
    try:
        tmp = Image.open('../Dataset/Vietnam_map/negative/english/'+n)
        if tmp.size[0]==0:
            breakpoint()
    except:
        breakpoint()
