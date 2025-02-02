
import os, sys
from config.config1 import Config
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')

hyperparameters = Config()
os.makedirs(hyperparameters.CACHE_DIR, exist_ok = True)
