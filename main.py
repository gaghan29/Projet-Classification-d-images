from dataset import MelonaDataset
from transforms import transform_base, transform_normalise, MEAN, STD
from model import SimpleCNN, compter_parametres
from train import train_one_epoch, evaluate

import torch
