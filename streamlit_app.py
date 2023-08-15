import urllib.request 
from typing import List, Optional
import matplotlib.pyplot as plt
from matplotlib import patches

import numpy as np
import io

from PIL import Image
import torch
import torchvision as tv
from torchvision import transforms as T

from funcs import plot_bboxes
import streamlit as st

url = 'https://github.com/GrigoryMaslov/BarkBeetle-RetinaNet/releases/download/v1.0.0/RetinaNetDD_40'
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)

damage_detection_model = tv.models.detection.retinanet_resnet50_fpn_v2(
                                  weights=None,
                                  nms_thresh = 0.5,
                                  topk_candidates = 200,
                                  detections_per_img = 100,
                                  score_thresh = 0.15,
                                  num_classes=2)

damage_detection_model.load_state_dict(torch.load('RetinaNetDD_40', map_location=torch.device('cpu') ))

def get_transform():
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float32))
    return T.Compose(transforms)

transforms = get_transform()

def main():
  execute_recsys=False
  img = Image.open("sample_image.jpg").convert("RGB")
  st.markdown("# Initial image: ")
  st.image(img)
  execute_recsys = st.button("Detect bark beetle!")
  if execute_recsys:
    img_transformed = transforms(img)
    damage_detection_model.eval()
    with torch.no_grad():
      preds = damage_detection_model(img_transformed.unsqueeze(0))
    labels = ['Damage!' for pred in preds[0]['labels']]
    damage = preds[0]['labels']
    bboxes = preds[0]['boxes']
    result = plot_bboxes(img=img, bboxes=bboxes,labels=labels, damage=damage)
    img_buf = io.BytesIO()
    result.savefig(img_buf, format='png')
    image_predicted = Image.open(img_buf)
    st.markdown("# Prediction: ")
    st.image(image_predicted)
    execute_recsys=False

while True:
  main()
