import random
import numpy as np
import torch
from PIL import Image
from torch.hub import load_state_dict_from_url
import os

# Convert image to RGB mode
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        return image.convert('RGB')

# Resize image and pad
def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w/iw, h/ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh

# Get the current learning rate of the optimizer
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Set random seeds to ensure repeatability of experimental results
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Initialize the DataLoader worker
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# Preprocess the input image to normalize pixel values ​​to the [0, 1] range
def preprocess_input(image):
    return image / 255.0

# Print configuration information
def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

# Download pre-trained weights
def download_weights(backbone, model_dir="./model_data"):
    download_urls = {
        'mobilenet': 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
    }
    url = download_urls.get(backbone)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    load_state_dict_from_url(url, model_dir)
