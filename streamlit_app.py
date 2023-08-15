import urllib.request 
from typing import List, Optional
import matplotlib.pyplot as plt
from matplotlib import patches

import numpy as np

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

def main():
  image = Image.open("sample_image.jpg")
  st.image(image)
  st.markdown("# All done!")

main()
