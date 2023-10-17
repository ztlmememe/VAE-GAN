# The goal of this program is to, using pytorch:
# 1. Take in sample pairs of real images and their respective distilled images
# 2. Train a DiffAug model to regenerate the distilled image given a distelled image
# 3. Finetune the DiffAug encoder to generate latents that can make 
#    distilled images given real images (MSE with distilled image as target)
# 4. Train a CNN on the new distilled images

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from models import *
from utils import *



# Parse arguments
def main():
    pass
