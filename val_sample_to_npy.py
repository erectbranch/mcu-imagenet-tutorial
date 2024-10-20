import argparse
import os
import yaml

import numpy as np
import torch

from utils.configs import SamplingConfig, yaml_to_config
from utils.dataset import build_dataset
from utils.sampler import build_sampler, get_subset_info, save_subset_info

def setup_args():
    parser = argparse.ArgumentParser('ImageNet validation dataset to .npy format script', add_help=False)
    parser.add_argument('config', metavar="FILE", help='sampling config file (./configs/sampling.yaml)')
    parser.add_argument('--not_permute', action='store_true', default=False,  help='pytorch to tflite input dim sequence')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--verbose', action='store_true', help='save sample image path')
    args = parser.parse_args()
    
    return args


def image_to_npz(data_loader, config, not_permute:bool=False):
    save_path = config.sampling.save_path
    os.makedirs(save_path, exist_ok=True)

    subset_index = 1
    image_list = []
    target_list = []

    for i, (images, target) in enumerate(data_loader):
        if not not_permute:
            images = images.permute((0, 2, 3, 1))      # N, C, H, W -> N, H, W, C
        
        if len(target.shape) == 1:
            new_target = torch.zeros((target.shape[0], 1000))
            new_target[torch.arange(target.shape[0]), target] = 1
            target = new_target
        
        image_list.append(images)
        target_list.append(target)

        if (i+1) % config.sampling.subset_size == 0:
            np.save(f'{os.path.join(save_path, f"{subset_index}_input.npy")}',  [x.numpy() for x in image_list],  allow_pickle=True)
            np.save(f'{os.path.join(save_path, f"{subset_index}_target.npy")}', [y.numpy() for y in target_list], allow_pickle=True)

            subset_index += 1
            image_list = []
            target_list = []


if __name__ == '__main__':
    args = setup_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = yaml_to_config(config, SamplingConfig)

    val_dataset = build_dataset(config)
    val_sampler, chosen_indexes = build_sampler(config, length=len(val_dataset))

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        sampler=val_sampler, 
        batch_size=config.valid_set.batch_size, 
        num_workers=args.num_workers
    )

    image_to_npz(val_dataloader, config, not_permute=args.not_permute)

    if args.verbose:
        assert config.valid_set.path.endswith('val/') or config.valid_set.path.endswith('val'), 'Please check the path path of the validation dataset'
        
        subset_img_list = get_subset_info(chosen_indexes, val_dataset, valid_dir='val/')
        save_subset_info(subset_img_list, save_path=os.path.join(config.sampling.save_path, 'subset_info.txt'))
    