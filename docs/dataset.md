
# VinMap dataset

# Map classification
## Classification 1: Map classification
## Classification 2: Vietnam Map classification
# ICDAR 2015 pretraining
In this section ICDAR 2015 pretraining for Text detection and Text recognition are presented. The same procedure is applied for VinText dataset!
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
    "isArt": false
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
    "isArt": false
  },
]
```

![Alt text](visualization/polygon_interpolation.png?raw=true "Polygon")

We interpolate multiple vertices polygon into 4 vertices polygon for easy segmentation.
The .txt text ICDAR2015 style format for each image polygon (4 vertices):
```python
228,58,51,58,51,262,227,262,Ô
891,47,279,27,269,299,881,319,KÌA!
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