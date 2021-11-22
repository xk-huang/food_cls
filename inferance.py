import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

import torch.optim
import torchvision
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset

from dataset import Food_LT
from model import resnet34
import config as cfg
from utils import adjust_learning_rate, save_checkpoint, train, validate, logger, load_checkpoint
import time
import os.path as osp
from datetime import datetime
from tqdm import tqdm


def main():
    model = resnet34()

    print('log save at:' + cfg.log_path)
    
    if not torch.cuda.is_available():
        os._exit(0)

    if cfg.gpu is not None:
        print('Use cuda !')
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)

    print('Load dataset ...')
    dataset = Food_LT(False, root=cfg.root, batch_size=cfg.batch_size, num_works=4)
    test_loader = dataset.test

    if cfg.resume:
        ''' plz implement the resume code by ur self! '''

        ckpt = load_checkpoint(cfg.root, model_type='best')
        if ckpt is not None:
            # start_epoch = ckpt['epoch']
            model.load_state_dict(ckpt['state_dict_model'])
        print("load ckpt")

    all_outputs = []
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm(enumerate(test_loader)):
            if cfg.gpu is not None:
                images = images.cuda(cfg.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(cfg.gpu, non_blocking=True)


            output = model(images)
            output = output.argmax(-1)
            all_outputs.append(output.cpu().numpy())
    all_outputs = np.concatenate(all_outputs)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    with open(cfg.root + 'log/' + f'submission.{dt_string}.txt', 'w') as f:
        f.write('Id,Expected\n')
        for i, j in zip(dataset.test_names, all_outputs):
            f.write(f'{osp.basename(i)},{j}\n')
            

    print('Finish !')


if __name__ == '__main__':
    main()