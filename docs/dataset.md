
# VinMap dataset
**Dataset link:** [Google Drive](https://drive.google.com/drive/folders/1Pgzv6hz977c82HpeSZsgkY5CYxG5FRV2?usp=sharing)
* MapData (Classification#1)
* MapT12_classification (Classification#2)
* Maps containing HS TS - Box Labled
* VinMap

Download and unzip the files. For classification, the dataset is already formatted for training/testing. For detection/recognition, follow bellow steps for pretraining and fine-tuning the model.

# VinMap dataset 
The dataset policy (positive/negative) is determined by the our paper.

    VinMap
    │   ├── positive
    │   │   ├── english
    │   │   ├── vietnamese
    │   ├── negative
    │   │   ├── english
    │   │   ├── vietnamese

Box annotations labeling process: [Youtube link](https://www.youtube.com/watch?v=R_eyNzhdJuE)

    Maps containing HS TS - Box Labled
    │   ├── Vietnamese
    │   │   ├── english
    │   │   ├── vietnamese
    │   ├── Not Vietnamese
    │   │   ├── english
    │   │   ├── vietnamese
# Map classification
## Classification 1: Map classification

Classify if image is a map or not

    MapData
    │   ├── map
    │   │   ├── anhbando.png
    │   │   ├── map.jpg
    │   │   ├── ...
    │   ├── notmap
    │   │   ├── coco.jpg
    │   │   ├── car.png
    │   │   ├── ...
## Classification 2: Vietnam Map classification

Classify if the map is Vietnam map or not

    MapT12_classification
    │   ├── vietnam
    │   │   ├── bandovietnam.png
    │   │   ├── hoangsatruongsa.jpg
    │   │   ├── ...
    │   ├── notvietnam
    │   │   ├── russia.jpg
    │   │   ├── seattle_usa.png
    │   │   ├── ...

# ICDAR 2015 pretraining
In this section ICDAR 2015 pretraining for Text detection and Text recognition are presented. The same procedure is applied for VinText and VinMap dataset.
## Text Detection
For the detection task, we provide the data conversion from json to ICDAR2015 .txt style.
The json dictionary format for each image polygon (multiple vertices):
```python
[
# Vietnamese encoder 
  {
    "text": "\u00d4",
    "points": [
      [52, 59],
      [228, 59],
      [228,262],
      [52,262]],
  },
# Unlabeled text   
  {
    "text": "###",
    "points": [
      [52, 59],
      [228, 59],
      [234, 62],
      [256, 65],
      [278, 66],
      [228,64],
      [52,262]],
  },
]
```

We interpolate multiple vertices polygon into 4 vertices polygon for easy segmentation.
The .txt text ICDAR2015 style format for each image polygon (4 vertices):
```python
228,58,51,58,51,262,227,262, HOANG
891,47,279,27,269,299,881,319,SA!
```
In order to change the annotation to ICDAR2015 (including interpolating the muti-vertice polygon and cleaning for unlabeled text), proceed the following code
```python
python tools/data_converter.py --label_root <root_anno_labels> --label_des <converted_output_anno_labels>
```
Ensuring the right data tree format

    Home
    ├── Dataset
    │   ├── images
    │   │   ├── training
    │   │   ├── validation
    │   │   ├── testing
    │   ├── labels
    │   │   ├── training
    │   │   ├── validation
    │   │   ├── testing

After having converted to the ICDAR2015, we will use the configuration of MMOCR+MMDetection to train the text detector using the COCO format. We also provide the conversion to COCO format:
```python
python tools/convert_annotation.py --image_path Dataset/images --gt_path Dataset/labels --out-dir Dataset/json_converted --dataset icdar2015 --split-list training
python tools/convert_annotation.py --image_path Dataset/images --gt_path Dataset/labels --out-dir Dataset/json_converted --dataset icdar2015 --split-list validation
python tools/convert_annotation.py --image_path Dataset/images --gt_path Dataset/labels --out-dir Dataset/json_converted --dataset icdar2015 --split-list testing
```
Desired outcome

    Home
    ├── Dataset
    │   ├── images
    │   │   ├── training
    │   │   ├── validation
    │   │   ├── testing
    │   ├── labels
    │   │   ├── training
    │   │   ├── validation
    │   │   ├── testing
    │   ├── json_converted
    │   │   ├── instances_training.json
    │   │   ├── instances_validation.json
    │   │   ├── instances_testing.json

## Text Recognition
Crop the bounding boxes out as your wish. Modifying your datatree
    
    vietocr
    ├── data
    │   ├── images
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   ├── ...
    │   │   ├── 3.png
    │   │   ├── annotation.txt
    
The annotation.txt folows the format of (No 2 words per image):
```python
data/images/1.png Ford
data/images/2.png PARA
data/images/3.png JUNIPER
data/images/4.png TABLE
...
```