import urllib.request 
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
  st.markdown("# Bark beetle damage detection")
  st.markdown("## With RetinaNet")
  st.markdown("This page showcases the model for bark beetle detection. RetinaNet uses [$FL = -(1-P_t)^\gamma ln(P_t)$]")
  
  retrieve_image = {'Backsjon, Sweden': 'images/backsjon_vertical.jpg',
                   'Lidhem, Sweden': 'images/lidhem_oblique.jpg',
                   'Just some illustrative image from the Internet': 'images/sample_image.jpg'}
  
  keys = list(retrieve_image.keys())
  
  option = st.selectbox(
    'Where are we going to look for bark beetle?',
    (keys[0], keys[1], keys[2]))
  
  img = Image.open(retrieve_image[option]).convert("RGB")
  st.image(img)
  
  score_threshold = st.slider('Choose the score threshold for the object detection model: ', min_value=0.0, max_value=1.0, value=0.15, step=0.05)
  execute=False
  execute = st.button("Detect bark beetle!")
  if execute:
    img_transformed = transforms(img)
    damage_detection_model.score_thresh = score_threshold
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

main()
