import numpy as np
import torch

from torch.utils.data.sampler import SubsetRandomSampler

def setup_seed(manual_seed: int):
    import random
    torch.manual_seed(manual_seed)
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manual_seed) 


def build_sampler(config, length: int):
    setup_seed(config.sampling.random_seed)
    total_subset_size = config.sampling.subset_size * config.sampling.num_subset
    chosen_indexes = np.random.choice(list(range(length)), total_subset_size)
    
    val_sampler = SubsetRandomSampler(chosen_indexes)
    # val_sampler = torch.utils.data.SequentialSampler(val_dataset)     # If you want to save all images

    return val_sampler, chosen_indexes


def get_subset_info(chosen_indexes: list, val_dataset, valid_dir: str):
    img_list = []
    for i in chosen_indexes:
        img_path = val_dataset.imgs[i][0].split(valid_dir)[-1].split('/')
        image_folder = img_path[0]
        image_name   = img_path[1]

        img_list.append(f'{image_folder}/{image_name}')    
        
    return img_list


def save_subset_info(img_list: list, save_path: str):
    with open(save_path, 'w') as f:
        for img in img_list:
            f.write(f'{img}\n')