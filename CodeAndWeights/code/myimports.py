import os
import sys
import time
import copy
import glob
import random
import datetime
import argparse
import warnings
import itertools
from collections import OrderedDict
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast

import torchvision.transforms.functional as VF
from torchvision import transforms

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_fscore_support,
    f1_score, accuracy_score,
    precision_score, recall_score,
    balanced_accuracy_score
)

from aeon.datasets import load_classification
from timm.optim.adamp import AdamP

from utils import *
from mydataload import loadorean
from lookhead import Lookahead
from models.Itime import ItimeNet

# Suppress all warnings
warnings.filterwarnings("ignore")
