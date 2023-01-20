## 라이브러리 추가하기

import argparse

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import *
from util import *

# Parser 생성하기

parser = argparse.ArgumentParser(description="Regression Tasks such as inpainting, denoising, and super_resolution",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./content/drive/MyDrive/002-pytorch-image-regression-framework/datasets/BSR/BSDS500/data/images", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./content/drive/MyDrive/002-pytorch-image-regression-framework/checkpoint/inpainting/plain", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./content/drive/MyDrive/002-pytorch-image-regression-framework/log/inpainting/plain", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./content/drive/MyDrive/002-pytorch-image-regression-framework/result/inpainting/plain", type=str, dest="result_dir")

parser.add_argument("--task", default="super_resolution", choices=["inpainting", "denoising", "super_resolution"], type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=['bilinear', 4], dest='opts')

parser.add_argument("--ny", default=320, type=int, dest="ny")
parser.add_argument("--nx", default=480, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")
# image 사이즈 가변 조절(y, x, channel, unet kernel)

parser.add_argument("--network", default="unet", choices=["unet", "hourglass"], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

# network 설정 argument

args = parser.parse_args()

## 트레이닝 파라미터 설정
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
# 훈련된 네트워크 저장될 checkpoint 디렉토리
log_dir = args.log_dir
# 텐서보드 로그 파일 디렉토리
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float64)]

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker

network = args.network
learning_type = args.learning_type

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 디바이스 설정

print('mode: %s' % mode)

print('learning rate: %.4e' % lr)
print('batch size: %d' % batch_size)
print('number of epoch: %d' % num_epoch)

print('data dir: %s' % data_dir)
print('ckpt dir: %s' % ckpt_dir)
print('log dir: %s' % log_dir)
print('result dir: %s' % result_dir)

print('task: %s' % task)
print('opts: %s' % opts)

print('network: %s' % network)
print('learning type: %s' % learning_type)


#디렉토리 생성

result_dir_train = os.path.join(result_dir, 'train')
result_dir_val = os.path.join(result_dir, 'val')
result_dir_test = os.path.join(result_dir, 'test')

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir_train, 'png'))
    os.makedirs(os.path.join(result_dir_val, 'png'))

    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'numpy'))

## 네트워크 학습

if mode == 'train':
    transform_train = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5), RandomFlip()])
    transform_val = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5)])

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    # train data

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform_val, task=task, opts=opts)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)
    # validation data

    # 부수적인 variable 설정
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform_test = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5)])
    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_test, task=task, opts=opts)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    # 부수적인 variable 설정
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)


## 네트워크 생성
if network == 'unet':
    net = UNet(nch=nch, nker=nker, norm='bnorm', learning_type=learning_type).to(device)
# elif network == 'resnet':
# net = ResNet().to(device)


# loss function
#fn_loss = nn.BCEWithLogitsLoss().to(device)
# BCE - Segmentation 수행 시 손실함수

fn_loss = nn.MSELoss().to(device)
# image regression & restoration(복원) 수행 시 손실함수

# optimizer
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 부수적인 function 설정
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)
cmap = None

# denormalization 함수

# fn_class = lambda x: 1.0 * (x > 0.5)
# output 이미지를 binary 클래스로 분류해주는 함수

## tensorboard 사용을 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))


## 네트워크 학습시키기

st_epoch = 0
# 트레이닝이 시작되는 epoch position 을 0 으로 설정

if mode == 'train':
    # train mode
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
        # 네트워크 학습 이전에 저장돼있는 네트워크 있다면, 로드 후 연속적으로 네트워크 학습시킬수 있게 구현

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_mse = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # 손실함수 계산
            loss_mse += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_mse)))

            # Tensorboard 저장
            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            # matplotlib 이미지 save 시, 원활한 이미지 저장을 위해 matrix range 0 ~ 1 로 clipping
            input = np.clip(input, a_min=0, a_max=1)
            output = np.clip(output, a_min=0, a_max=1)

            id = num_batch_train * (epoch - 1) + batch

            plt.imsave(os.path.join(result_dir_train, 'png', '%04d_label.png' % id), label[0])
            plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input.png' % id), input[0])
            plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0])

            # writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_mse), epoch)


    ## Network validation 하는 부분
    # validation - back propagation 영역이 없기 때문에, torch.no_grad 로 사전에 방지

        with torch.no_grad():
            net.eval()
            # validation 위해 eval function 사용
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # loss function 계산
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print('VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f' %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorboard 저장
                label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

                input = np.clip(input, a_min=0, a_max=1)
                output = np.clip(output, a_min=0, a_max=1)

                id = num_batch_val * (epoch - 1) + batch

                plt.imsave(os.path.join(result_dir_val, 'png', '%04d_label.png' % id), label[0])
                plt.imsave(os.path.join(result_dir_val, 'png', '%04d_input.png' % id), input[0])
                plt.imsave(os.path.join(result_dir_val, 'png', '%04d_output.png' % id), output[0])

                # writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                # writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                # writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 50 == 0:
            # epoch 50 회 수행할때마다 해당 네트워크 저장
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

# test mode
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
    with torch.no_grad():
        net.eval()
        # validation 위해 eval function 사용
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # loss function 계산
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print('TEST: BATCH %04d / %04d | LOSS %.4f' %
                  (batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard 저장
            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            for j in range(label.shape[0]):
                id = batch_size * (batch - 1) + j

                label_ = label[j]
                input_ = input[j]
                output_ = output[j]

                np.save(os.path.join(result_dir_test, 'numpy', '%04d_label.npy' % id), label_)
                np.save(os.path.join(result_dir_test, 'numpy', '%04d_input.npy' % id), input_)
                np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_)

                label_ = np.clip(label_, a_min=0, a_max=1)
                input_ = np.clip(input_, a_min=0, a_max=1)
                output_ = np.clip(output_, a_min=0, a_max=1)

                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_label.png' % id), label_, cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input.png' % id), input_, cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=cmap)

    print('AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f' % (batch, num_batch_test, np.mean(loss_arr)))
