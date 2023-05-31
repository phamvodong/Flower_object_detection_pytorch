import torch

BATCH_SIZE = 2      # increase / decrease according to the GPU memory
NUM_EPOCHS = 50      # number of epochs to for

# Identify the device for model
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Labels in the flower dataset
LABEL_MAP = {'Babi': 1, 'Calimerio': 2, 'Chrysanthemum': 3, 'Hydrangeas': 4, 'Lisianthus': 5, 'PingPong': 6, 'Rosy': 7, 'Tana': 8}

# The number of classes in the dataset
NUM_CLASSES = 9

# Training dataset folder
TRAIN_DIR = '../data/meta_data/train/train.csv'

# Validation dataset folder
VALID_DIR = '../data/meta_data/val/val.csv'

# Flower dataset
FLOWER_DIR = '../data/Flowers'

# Out put direction
OUT_DIR = 'output/'

# Num workers
NUM_WORKERS = 2

# Checkpoint interval
CHECKPOINT_INTERVAL = 1