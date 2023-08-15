import urllib.request 
from typing import List, Optional
import matplotlib.pyplot as plt
from matplotlib import patches

import numpy as np

from PIL import Image
import torch
import torchvision as tv
from torchvision import transforms as T

url = 'https://github.com/GrigoryMaslov/BarkBeetle-RetinaNet/releases/download/v1.0.0/RetinaNetDD_40'
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)
