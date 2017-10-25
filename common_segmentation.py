import sys

import torch
from torch.autograd import Variable
from torch.optim import Adam, SGD

from models.u_net_1_channel import UNet
from data import RetinalVesselTrainingDS
from utils.misc import CrossEntropyLoss2d

from piwise.visualize import Dashboard
from piwise.transform import Colorize

import argparse
import math

def parse_args():
    parser = argparse.ArgumentParser(description='vessel segmentation')
    parser.add_argument('--root', required=True)
    parser.add_argument('--patch_size', default=256, type=int)
    parser.add_argument('--model', default='unet', choices=['unet', 'pspnet', 'fcn8', 'fcn16',
                                                            'fcn32', 'duc_hdc', 'gcn'])
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--batch', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--mom', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--phase', default='train', choices=['train', 'test'])
    parser.add_argument('--display', default=100, type=int)
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--output', default='output', help='The output dir')
    parser.add_argument('--weight', default=None)
    return parser.parse_args()

model = UNet(2)

data_root = './DRIVE_datasets_training_testing/'
img_file = 'DRIVE_dataset_imgs_train.hdf5'
mask_file = 'DRIVE_dataset_groundTruth_train.hdf5'
patch_w = args.batch
patch_h = args.batch
patch_num_per_img = math.ceil((2048/args.patch_size)*(2048/args.patch_size))
data_set = RetinalVesselTrainingDS(data_root, img_file, mask_file,
                                      patch_w, patch_h, patch_num_per_img)
data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=10, shuffle=True, pin_memory=True)

criterion = CrossEntropyLoss2d()
optimizer = SGD(model.parameters(), 1e-4, .9, 2e-5)

board = Dashboard('8097')
color_transform = Colorize()


def train(train_dataloader, model, criterion, optimizer, epoch, display):
    model.train()

    for index, (img, mask) in enumerate(train_dataloader):
        input = Variable(img.cuda())
        target = Variable(mask.type(torch.LongTensor).cuda())
        output = model(input)
        optimizer.zero_grad()
        loss = criterion(output, target[:, 0])
        loss.backward()
        optimizer.step()
        print('loss: {}'.format(loss.cpu().data[0]))
        if index % 100 == 0:
            image = img[0]
            image_rgb = torch.FloatTensor(3, image.shape[1], image.shape[2])
            image_rgb[0] = image[0] * .229 + .485
            image_rgb[1] = image[0] * .224 + .456
            image_rgb[2] = image[0] * .225 + .406
            board.image(image_rgb,
                        f'input (epoch: {epoch}, step: {index})')
            board.image(color_transform(output[0].cpu().max(0)[1].data.unsqueeze(0)),
                        f'output (epoch: {epoch}, step: {index})')
            board.image(color_transform(target[0].cpu().data),
                        f'target (epoch: {epoch}, step: {index})')



def test_train():
    for i in range(200):
        train(data_loader, torch.nn.DataParallel(model).cuda(), criterion, optimizer, i, 10)

test_train()




