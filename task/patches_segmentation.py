# This file used to support template for common segmentation task
# The input data in this case need to be prepared with scale and ahe operation

# import
# sys
import sys
sys.path.append('../')
sys.path.append('../models')
import os
# utils
import argparse
import time
from glob import glob
#torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torch.optim as optim
#torchvision
import torchvision
import torchvision.transforms as transforms
#local lib model
from models.u_net import UNet
from models.duc_hdc import ResNetDUC, ResNetDUCHDC
from models.fcn32s import FCN32VGG
from models.fcn16s import FCN16VGG
from models.fcn8s import FCN8s
from models.gcn import GCN
from models.psp_net import PSPNet
from models.seg_net import SegNet
from piwise.network import FCN8
from torch.optim import Adam
#local lib dataset
from dataset.vessel_ahe_dataset import RetinalVesselTrainingDS, RetinalVesselValidationDS, \
    RetinalVesselPredictImage, recompone_overlap
from dataset.patches_ahe_dataset import patch_input_transform, patch_label_transform, PatchesTrainingDS
# local lib utils
from utils.misc import CrossEntropyLoss2d
from utils.utils import AverageMeter
# visualization&local lib: piwise
from piwise.visualize import Dashboard
from piwise.transform import Colorize

# math related
import numpy as np

# for test
import cv2


# globe scope
color_transform = Colorize()

def parse_args():
    parser = argparse.ArgumentParser(description='patches segmentation task')
    parser.add_argument('--root', required=True)
    parser.add_argument('--root_val', required=True)
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--patch_size', default=256, type=int)
    parser.add_argument('--model', default='unet', choices=[
        'unet', 'fcn8', 'fcn16', 'fcn32', 'gcn', 'pspnet',
        'duc', 'duc_hdc', 'segnet', 'fcn8p'
    ])
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--port', default=8097, type=int)
    parser.add_argument('--epoch', default=2000, type=int)
    parser.add_argument('--batch', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--mom', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--optim', default='sgd', choices=[
        'sgd', 'adam', 'adadelta'
    ])
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--steps_plot', default=200, type=int)
    parser.add_argument('--steps_loss', default=200, type=int)
    parser.add_argument('--steps_save', default=300, type=int)
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--phase', default='train', choices=[
        'train',
        'test',
        'infer',
        'predict'
    ])
    parser.add_argument('--output', default='output', help='The output dir')
    parser.add_argument('--weight', default=None)
    parser.add_argument('--dataset', default='common')
    parser.add_argument('--exp', default='patches_segmentation')
    parser.add_argument('--fix', default=500, type=int)
    parser.add_argument('--step', default=500, type=int)
    return parser.parse_args()

def get_model(model_name, num_classes, weight=None):
    print('====> load {}-classes segmentation model: {}'.format(num_classes, model_name))
    model = None
    if model_name == 'fcn8':
        model = FCN8s(num_classes=num_classes)
    elif model_name == 'fcn8p':
        model = FCN8(num_classes=num_classes)
    elif model_name == 'fcn16':
        model = FCN16VGG(num_classes=num_classes, pretrained=False)
    elif model_name == 'fcn32':
        model = FCN32VGG(num_classes=num_classes, pretrained=False)
    elif model_name == 'unet':
        model = UNet(num_classes=num_classes)
    elif model_name == 'duc':
        model = ResNetDUC(num_classes=num_classes)
    elif model_name == 'duc_hdc':
        model = ResNetDUCHDC(num_classes=num_classes)
    elif model_name == 'gcn':
        model = GCN(num_classes=num_classes, input_size=512)
    elif model_name == 'pspnet':
        model = PSPNet(num_classes=num_classes)
    elif model_name == 'segnet':
        model = SegNet(num_classes=num_classes)
    if weight is not None:
        model.load_state_dict(torch.load(weight))
    return model

def test_get_model():
    num_classes = 3
    model_name = 'fcn8'
    model = get_model(model_name=model_name, num_classes=num_classes)
    model_name = 'fcn16'
    model = get_model(model_name=model_name, num_classes=num_classes)
    model_name = 'fcn32'
    model = get_model(model_name=model_name, num_classes=num_classes)
    model_name = 'unet'
    model = get_model(model_name=model_name, num_classes=num_classes)
    model_name = 'duc'
    model = get_model(model_name=model_name, num_classes=num_classes)
    model_name = 'duc_hdc'
    model = get_model(model_name=model_name, num_classes=num_classes)
    model_name = 'gcn'
    model = get_model(model_name=model_name, num_classes=num_classes)
    model_name = 'pspnet'
    model = get_model(model_name=model_name, num_classes=num_classes)
    model_name = 'segnet'
    model = get_model(model_name=model_name, num_classes=num_classes)

def get_optimizer(optim_name):
    optimizer = optim.SGD
    if optim_name == 'sgd':
        optimizer = optim.SGD
    elif optim_name == 'adadelta':
        optimizer == optim.Adadelta
    elif optim_name == 'adam':
        optimizer = optim.Adam
    elif optim_name == 'adagrad':
        optimizer = optim.Adagrad
    elif optim_name == 'RMSprop':
        optimizer = optim.RMSprop
    return optimizer

# import cv2
# import numpy as np

def train(train_dataloader, model, criterion, optimizer, epoch,
          steps_plot=None, steps_loss=None, steps_save=None,
          board=None):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    logger = []
    for step, (imgs, masks) in enumerate(train_dataloader):
        data_time.update(time.time()-end)
        input = Variable(imgs.cuda())
        target = Variable(masks.type(torch.LongTensor).cuda())
        output = model(input)
        optimizer.zero_grad()
        loss = criterion(output, target.squeeze())
        loss.backward()
        optimizer.step()
        batch_time.update(time.time()-end)
        losses.update(loss.cpu().data[0], len(imgs))
        if steps_loss is not None:
            if step%steps_loss == 0 and steps_loss > 0:
                print_info = 'Epoch: [{epoch}][{step}/{tot}]\t' \
                             'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                             'Data: {data_time.avg:.3f}\t' \
                             'Loss: {loss.avg:.4f}\t'.format(
                    epoch = epoch,
                    step = step,
                    tot = len(train_dataloader),
                    batch_time = batch_time,
                    data_time = data_time,
                    loss = losses
                )
                print(print_info)
                logger.append(print_info)
        if steps_plot is not None and board is not None:
            if step%steps_plot == 0 and steps_plot > 0:
                # plot image on visdom server
                image = imgs[0]
                image[0] = image[0] * .229 + .485
                image[1] = image[1] * .224 + .456
                image[2] = image[2] * .225 + .406
                board.image(image,
                            f'input (epoch: {epoch}, step: {step})')
                board.image(color_transform(output[0].cpu().max(0)[1].data.unsqueeze(0)),
                            f'output (epoch: {epoch}, step: {step})')
                board.image(color_transform(target[0].cpu().data),
                            f'target (epoch: {epoch}, step: {step})')
                # cv_img = np.array(transforms.ToPILImage()(masks[0]))
                # cv2.imshow('test', cv_img)
                # cv2.waitKey(10000)
    return logger, losses.avg

def val(val_dataloader, model, criterion, epoch,
          steps_plot=None, steps_loss=None, steps_save=None,
          board=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    logger = []
    for step, (imgs, masks) in enumerate(val_dataloader):
        data_time.update(time.time()-end)
        input = Variable(imgs.cuda())
        target = Variable(masks.type(torch.LongTensor).cuda())
        output = model(input)
        loss = criterion(output, target.squeeze())
        batch_time.update(time.time()-end)
        losses.update(loss.cpu().data[0], len(imgs))
        print_info = 'Eval: [{epoch}][{step}/{tot}]\t' \
                     'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                     'Data: {data_time.avg:.3f}\t' \
                     'Loss: {loss.avg:.4f}\t'.format(
            epoch=epoch,
            step=step,
            tot=len(val_dataloader),
            batch_time=batch_time,
            data_time=data_time,
            loss=losses
        )
        print(print_info)
        logger.append(print_info)
        image = imgs[0]
        image[0] = image[0] * .229 + .485
        image[1] = image[1] * .224 + .456
        image[2] = image[2] * .225 + .406
        # board.image(image,
        #             f'input (epoch: {epoch}, step: {step})')
        # board.image(color_transform(output[0].cpu().max(0)[1].data.unsqueeze(0)),
        #             f'output (epoch: {epoch}, step: {step})')
        # board.image(color_transform(target[0].cpu().data),
        #             f'target (epoch: {epoch}, step: {step})')
        # cv_img = np.array(transforms.ToPILImage()(masks[0]))
        # cv2.imshow('test', cv_img)
        # cv2.waitKey(10000)
    return logger, losses.avg

def pred_image(image_file, pred_image_dataloader, model, size, patch_size, stride, board):
    model.eval()
    pred_image_patches = torch.FloatTensor(len(pred_image_dataloader.dataset), 1, patch_size, patch_size)
    patches_cnt = 0
    for index, (images) in enumerate(pred_image_dataloader):
        input = Variable(images.cuda())
        output = model(input)
        predit_o = output.cpu().data.max(1)[1].unsqueeze(1).type(torch.FloatTensor)
        for i in range(predit_o.shape[0]):
            pred_image_patches[patches_cnt,:] = predit_o[i]
            patches_cnt += 1
    pred_image_patches = pred_image_patches.numpy()
    pred_image = recompone_overlap(pred_image_patches, size, size, stride, stride)
    pred_image = np.transpose(pred_image[0], (1,2,0))
    pred_image *= 255
    pred_image = np.array(pred_image, dtype=np.uint8)

    from PIL import Image
    raw_img = Image.open(image_file)
    raw_img = np.array(raw_img, dtype=np.uint8)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)

    # stitch_img = np.empty((raw_img.shape[0] * 2, raw_img.shape[1], 3), dtype=np.uint8)
    # stitch_img[:raw_img.shape[0], :] = raw_img
    # pred_image = cv2.cvtColor(pred_image, cv2.COLOR_GRAY2RGB)
    # stitch_img[raw_img.shape[0]:, :] = pred_image
    stitch_img = np.empty((raw_img.shape[0], raw_img.shape[1]*2, 3), dtype=np.uint8)
    stitch_img[:,:raw_img.shape[0]] = raw_img
    pred_image = cv2.cvtColor(pred_image, cv2.COLOR_GRAY2RGB)
    stitch_img[:,raw_img.shape[0]:] = pred_image

    cv2.imshow('pred_image', stitch_img)
    # tmp_root = './data/retinal_vessel'
    # vessel_file = os.path.join(tmp_root, os.path.basename(image_file).split('.')[0].split('_')[0]+'.png')
    # cv2.imwrite(vessel_file, stitch_img)
    cv2.waitKey(4000)



def main():
    print('====> Retinal Image Lesions Segmentation: ')
    args = parse_args()
    print('====> Parsing Options: ')
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    output_dir = os.path.join(args.output,
                              args.dataset + '_lesions_segmentaion_' + args.phase + '_' + time_stamp + '_' + args.model + '_' + args.exp)
    if not os.path.exists(output_dir):
        print('====> Creating ', output_dir)
        os.makedirs(output_dir)
    print('====> load model: ')
    model = get_model(args.model, args.num_classes, args.weight)
    criterion = CrossEntropyLoss2d()
    # patches_per_image = 200
    print('====> start visualize dashboard: ')
    board = Dashboard(args.port)
    if args.phase == 'train':
        print('=====> Training model:')
        train_dataloader = DataLoader(PatchesTrainingDS(args.root, patch_input_transform, patch_label_transform),
                                       batch_size=args.batch,
                                       num_workers=args.workers,
                                       shuffle=True,
                                       pin_memory=True)

        val_dataloader = DataLoader(PatchesTrainingDS(args.root, patch_input_transform, patch_label_transform),
                                       batch_size=args.batch,
                                       num_workers=args.workers,
                                       shuffle=False,
                                       pin_memory=False)
        best_train_loss = 1e4
        best_val_loss = 1e4
        for epoch in range(args.epoch):
            if epoch < args.fix:
                lr = args.lr
            else:
                lr = args.lr * (0.1**(epoch//args.step))
            optimizer = get_optimizer(args.optim)
            optimizer = optimizer(model.parameters(), lr, args.mom, args.wd)
            optimizer = Adam(model.parameters())
            train_logger, train_loss = train(
                train_dataloader, nn.DataParallel(model).cuda(), criterion, optimizer, epoch,
                args.steps_plot, args.steps_loss, args.steps_save, board
            )
            # val_logger = []
            # val_loss = 0
            val_logger, val_loss = val(
                val_dataloader, nn.DataParallel(model).cuda(), criterion, epoch, board=board
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print_info = 'Current Validation Loss: {}'.format(best_val_loss)
                val_logger.append(print_info)
                print(print_info)
                tmp_file =os.path.join(output_dir, args.dataset+'_segmentation_'+args.model+'_%04d'%epoch+'_best.pth')
                print_info = '====> Save model: {}'.format(tmp_file)
                torch.save(model.cpu().state_dict(), tmp_file)
                val_logger.append(print_info)
                print(print_info)
            if epoch % 100 == 0:
                print_info = 'Current Validation Loss: {}\t\tCurrent Training Loss: {}'.format(best_val_loss, train_loss)
                val_logger.append(print_info)
                print(print_info)
                tmp_file = os.path.join(output_dir,
                                        args.dataset + '_segmentation_' + args.model + '_%04d' % epoch + '_best_record.pth')
                print_info = '====> Save model: {}'.format(tmp_file)
                torch.save(model.cpu().state_dict(), tmp_file)
                val_logger.append(print_info)
                print(print_info)
            if not os.path.isfile(os.path.join(output_dir, 'train.log')):
                with open(os.path.join(output_dir, 'train.log'), 'w') as fp:
                    fp.write(str(args)+'\n\n')
            with open(os.path.join(output_dir, 'train.log'), 'a') as fp:
                fp.write('\n' + '\n'.join(train_logger))
                fp.write('\n' + '\n'.join(val_logger))
    elif args.phase == 'predict':
        # image_file = '/home/weidong/code/dr/RetinalImagesVesselExtraction/data/DRIVE/test/ahe/02_test_ahe.png'
        # image_files = glob(os.path.join(args.root_val, 'ahe/*.png'))
        # image_files = glob(
        #     os.path.join('/home/weidong/code/github/DiabeticRetinopathy_solution/data/zhizhen_new/LabelImages/512_ahe',
        #                  '*.png'))

        # image_files = glob(os.path.join('/home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/e_optha_EX/ahe', '*.png'))
        image_files = glob(
            os.path.join('/home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/e_optha_EX/ahe_1024', '*.png'))
        for index in image_files:
            image_file = index
            size = 1024
            patch_size = 256
            stride = 256
            MEAN = [.485, .456, .406]
            STD = [.229, .224, .225]
            input_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)
                ]
            )
            color_trans = transforms.ToPILImage()
            ds = RetinalVesselPredictImage(image_file, input_transform, size, patch_size, stride)
            data_loader = DataLoader(ds, batch_size=20, shuffle=False, pin_memory=False)
            pred_image(image_file, data_loader, nn.DataParallel(model).cuda(), size, patch_size, stride, board)
    else    :
        raise Exception('No phase found')


if __name__ == '__main__':
    # test_get_model()
    main()