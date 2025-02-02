import os
import random
import numpy as np
import torch
from config.config1 import Config
hyperparameters = Config()

def seed_torch(seed=hyperparameters.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    seed_torch()
    print(hyperparameters)
