'''
/*
 * @Author: Jun Shi 
 * @Date: 2022-06-27 17:26:44 
 * @Last Modified by:   Jun Shi 
 * @Last Modified time: 2022-06-27 17:26:44 
 */
 '''
import os
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
import numpy as np
import math
import shutil
from tqdm import tqdm

from torch.nn import functional as F

import data_utils.transform as tr
from data_utils.data_loader import DataGenerator

from utils import dfs_remove_weight, hdf5_reader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
# GPU version.
import setproctitle
import copy

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomResizedCrop,
    RandomHorizontalFlip
)



class VolumeClassifier(object):
    '''
    Control the training, evaluation, and inference process.
    Args:
    - net_name: string, __all__ = ['r3d_18', 'mc3_18', 'r2plus1d_18',...].
    - lr: float, learning rate.
    - n_epoch: integer, the epoch number
    - channels: integer, the channel number of the input
    - num_classes: integer, the number of class
    - input_shape: tuple of integer, input dim
    - crop: integer, cropping size
    - batch_size: integer
    - num_workers: integer, how many subprocesses to use for data loading.
    - device: string, use the specified device
    - pre_trained: True or False, default False
    - weight_path: weight path of pre-trained model
    '''
    def __init__(self,
                 net_name=None,
                 lr=1e-3,
                 n_epoch=1,
                 channels=1,
                 num_classes=3,
                 input_shape=None,
                 crop=48,
                 scale=(-100, 200),
                 mean=None,
                 std=None,
                 use_roi=False,
                 temporal_frame=16,
                 batch_size=6,
                 num_workers=0,
                 device=None,
                 pre_trained=False,
                 external_pretrained_path=None,
                 weight_path=None,
                 weight_decay=0.,
                 momentum=0.95,
                 gamma=0.1,
                 milestones=[40, 80],
                 T_max=5,
                 use_fp16=True,
                 use_maxpool=False):
        super(VolumeClassifier, self).__init__()

        self.net_name = net_name
        self.lr = lr
        self.n_epoch = n_epoch
        self.channels = channels
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.crop = crop
        self.scale = scale
        self.mean = mean
        self.std = std
        self.use_roi = use_roi
        self.temporal_frame = temporal_frame

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        self.pre_trained = pre_trained
        self.external_pretrained_path = external_pretrained_path
        self.weight_path = weight_path
        self.start_epoch = 0
        self.global_step = 0
        self.loss_threshold = 1.0
        self.metric_threshold = 0.0
        # save the middle output
        self.feature_in = []
        self.feature_out = []

        self.weight_decay = weight_decay
        self.momentum = momentum
        self.gamma = gamma
        self.milestones = milestones
        self.T_max = T_max
        self.use_fp16 = use_fp16
        self.use_maxpool = use_maxpool

        os.environ['CUDA_VISIBLE_DEVICES'] = self.device

        self.net = self._get_net(self.net_name)

        if self.pre_trained:
            self._get_pre_trained(self.weight_path)
        
        
        if self.use_roi:
            self.transforms = [
                tr.Trunc_and_Normalize(self.scale, self.mean, self.std),
                tr.CropResize(dim=self.input_shape, crop=self.crop),
                tr.RandomTranslationRotationZoom(mode='trz'),
                # tr.RandomFlip(mode='v'),
                tr.To_Tensor()
            ]
        else:
            self.transforms = [
                ApplyTransformToKey(
                key="image",
                transform=Compose(
                    [
                        Lambda(lambda x: torch.from_numpy(x)),
                        UniformTemporalSubsample(self.input_shape[0]),
                        Lambda(lambda x: x / 255.0),
                        # Normalize(self.mean, self.std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        # RandomRCrop(self.input_shape[1]),
                        RandomResizedCrop(self.input_shape[1],scale=(0.75, 1.0), ratio=(0.75, 1.33)),
                        RandomHorizontalFlip(p=0.5),
                    ]
                    ),
                ),
                ]

    def trainer(self,
                train_path,
                val_path,
                label_dict,
                output_dir=None,
                log_dir=None,
                optimizer='Adam',
                loss_fun='Cross_Entropy',
                class_weight=None,
                lr_scheduler=None,
                cur_fold=0,
                repeat_factor=1.0):

        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        print('Device:{}'.format(self.device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        log_dir = os.path.join(log_dir, f'fold{str(cur_fold)}')
        output_dir = os.path.join(output_dir, f'fold{str(cur_fold)}')

        if os.path.exists(log_dir):
            if not self.pre_trained:
                shutil.rmtree(log_dir)
                os.makedirs(log_dir)
        else:
            os.makedirs(log_dir)

        if os.path.exists(output_dir):
            if not self.pre_trained:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)

        self.writer = SummaryWriter(log_dir)
        self.global_step = self.start_epoch * math.ceil(
            len(train_path) / self.batch_size)

        net = self.net
        lr = self.lr
        loss = self._get_loss(loss_fun, class_weight)

        if len(self.device.split(',')) > 1:
            net = DataParallel(net)

        # dataloader setting
        train_transformer = transforms.Compose(self.transforms)

        train_dataset = DataGenerator(train_path,
                                      label_dict,
                                      transform=train_transformer,
                                      use_roi=self.use_roi,
                                      repeat_factor=repeat_factor,
                                      temporal_frame=self.temporal_frame)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        # copy to gpu
        net = net.cuda()
        loss = loss.cuda()

        # optimizer setting
        optimizer = self._get_optimizer(optimizer, net, lr)
        scaler = GradScaler()

        # if self.pre_trained:
        #     checkpoint = torch.load(self.weight_path)
        # optimizer.load_state_dict(checkpoint['optimizer'])

        if lr_scheduler is not None:
            lr_scheduler = self._get_lr_scheduler(lr_scheduler, optimizer)

        early_stopping = EarlyStopping(patience=30,
                                       verbose=True,
                                       monitor='val_acc',
                                       best_score=self.metric_threshold,
                                       op_type='max')

        for epoch in range(self.start_epoch, self.n_epoch):
            setproctitle.setproctitle('{}: {}/{}'.format(
                'Shi Jun', epoch, self.n_epoch))
            train_loss, train_acc = self._train_on_epoch(
                epoch, net, loss, optimizer, train_loader, scaler)

            torch.cuda.empty_cache()

            val_loss, val_acc = self._val_on_epoch(epoch, net, loss, val_path,
                                                   label_dict)

            if lr_scheduler is not None:
                lr_scheduler.step()

            print('epoch:{},train_loss:{:.5f},val_loss:{:.5f}'.format(
                epoch, train_loss, val_loss))

            print('epoch:{},train_acc:{:.5f},val_acc:{:.5f}'.format(
                epoch, train_acc, val_acc))

            self.writer.add_scalars('data/loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('data/acc', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            self.writer.add_scalar('data/lr', optimizer.param_groups[0]['lr'],
                                   epoch)

            early_stopping(val_acc)

            if val_acc > self.metric_threshold:
                self.metric_threshold = val_acc

                if len(self.device.split(',')) > 1:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()

                saver = {
                    'epoch': epoch,
                    'save_dir': output_dir,
                    'state_dict': state_dict,
                    # 'optimizer': optimizer.state_dict()
                }

                file_name = 'epoch={}-train_loss={:.5f}-val_loss={:.5f}-train_acc={:.5f}-val_acc={:.5f}.pth'.format(
                    epoch, train_loss, val_loss, train_acc, val_acc)
                print('Save as :', file_name)
                save_path = os.path.join(output_dir, file_name)

                torch.save(saver, save_path)

            #early stopping
            if early_stopping.early_stop:
                print('Early Stopping for NO Improve!')
                break

            if self.metric_threshold == 1.0 and train_acc == 1.0:
                print('Early Stopping for Highest Metric!')
                break

        self.writer.close()
        dfs_remove_weight(output_dir, 3)

    def _train_on_epoch(self, epoch, net, criterion, optimizer, train_loader,
                        scaler):

        net.train()

        train_loss = AverageMeter()
        train_acc = AverageMeter()

        for step, sample in enumerate(train_loader):

            data = sample['image']
            target = sample['label']

            data = data.cuda()
            target = target.cuda()

            with autocast(self.use_fp16):
                output = net(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            if self.use_fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            output = F.softmax(output, dim=1)
            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            acc = accuracy(output.data, target)[0]
            train_loss.update(loss.item(), data.size(0))
            train_acc.update(acc.item(), data.size(0))

            torch.cuda.empty_cache()

            print(
                'epoch:{},step:{},train_loss:{:.5f},train_acc:{:.5f},lr:{:.5f}'
                .format(epoch, step, loss.item(), acc.item(),
                        optimizer.param_groups[0]['lr']))

            if self.global_step % 10 == 0:
                self.writer.add_scalars('data/train_loss_acc', {
                    'train_loss': loss.item(),
                    'train_acc': acc.item()
                }, self.global_step)

            self.global_step += 1

        return train_loss.avg, train_acc.avg

    def _val_on_epoch(self, epoch, net, criterion, val_path, label_dict):

        net.eval()

        val_transformer = transforms.Compose(self.transforms)

        val_dataset = DataGenerator(val_path,
                                    label_dict,
                                    transform=val_transformer,
                                    use_roi=self.use_roi,
                                    temporal_frame=self.temporal_frame)

        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True)

        val_loss = AverageMeter()
        val_acc = AverageMeter()

        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()

                with autocast(self.use_fp16):
                    output = net(data)
                    loss = criterion(output, target)

                output = F.softmax(output, dim=1)
                output = output.float()
                loss = loss.float()

                # measure accuracy and record loss
                acc = accuracy(output.data, target)[0]
                val_loss.update(loss.item(), data.size(0))
                val_acc.update(acc.item(), data.size(0))

                torch.cuda.empty_cache()

                print('epoch:{},step:{},val_loss:{:.5f},val_acc:{:.5f}'.format(
                    epoch, step, loss.item(), acc.item()))

        return val_loss.avg, val_acc.avg

    def hook_fn_forward(self, module, input, output):
        # print(module)
        # print(input[0].size())
        # print(output.size())

        for i in range(input[0].size(0)):
            self.feature_in.append(input[0][i].cpu().numpy())
            self.feature_out.append(output[i].cpu().numpy())

    def inference(self, test_path, net=None, hook_fn_forward=False):

        if net is None:
            net = self.net

        if hook_fn_forward:
            net.avgpool.register_forward_hook(self.hook_fn_forward)

        net = net.cuda()
        net.eval()

        test_transformer = transforms.Compose(self.transforms)

        test_dataset = DataGenerator(test_path,
                                     transform=test_transformer,
                                     use_roi=self.use_roi,
                                     temporal_frame=self.temporal_frame)

        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        result = {'pred': [], 'prob': []}

        with torch.no_grad():
            for _, sample in enumerate(test_loader):
                data = sample['image']
                data = data.cuda()
                with autocast(self.use_fp16):
                    output = net(data)
                output = F.softmax(output, dim=1)
                output = output.float()  #N*C

                result['pred'].extend(
                    torch.argmax(output, 1).detach().tolist())
                result['prob'].extend(output.detach().tolist())

                torch.cuda.empty_cache()

        return result, np.array(self.feature_in), np.array(self.feature_out)

    def inference_tta(self, test_path, tta_times=1, net=None):

        if net is None:
            net = self.net
        net = net.cuda()
        net.eval()

        test_transformer = transforms.Compose(self.transforms)

        prob_output = []
        vote_output = []
        with torch.no_grad():
            for item in tqdm(test_path):
                data = hdf5_reader(item, 'image')
                data = data.transpose((3,0,1,2))
                if self.use_roi:
                    data = data[0]
                    if data.shape[0] > self.temporal_frame:
                        depth = data.shape[0]
                        center = depth // 2
                        offset = self.temporal_frame//2
                        data = data[center - offset: center + offset]
                sample = {'image': data}
                # print(data.size)
                tta_output = []
                binary_output = []

                for _ in range(tta_times):
                    img_tensor = test_transformer(copy.deepcopy(sample))['image']
                    img_tensor = torch.unsqueeze(img_tensor, 0)

                    img_tensor = img_tensor.cuda()
                    with autocast(self.use_fp16):
                        output = net(img_tensor)
                    output = F.softmax(output, dim=1)
                    # print(output)
                    output = output.float().squeeze().cpu().numpy()
                    tta_output.append(output)

                    binary_output.append(np.argmax(output))

                prob_output.append(np.mean(tta_output, axis=0))
                vote_output.append(max(binary_output, key=binary_output.count))

        return prob_output, vote_output

    def _get_net(self, net_name):

        if net_name.startswith('r3d_') or net_name.startswith(
                'mc3_') or net_name.startswith('r2plus1d_'):
            import model.resnet_3d as resnet_3d
            net = resnet_3d.__dict__[net_name](input_channels=self.channels,
                                               num_classes=self.num_classes)

        elif net_name.startswith('da_'):
            import model.da_resnet_3d as da_resnet_3d
            net = da_resnet_3d.__dict__[net_name](input_channels=self.channels,
                                                  num_classes=self.num_classes)

        elif net_name.startswith('se_'):
            import model.se_resnet_3d as se_resnet_3d
            net = se_resnet_3d.__dict__[net_name](input_channels=self.channels,
                                                  num_classes=self.num_classes)

        elif net_name.startswith('vgg'):
            import model.vgg_3d as vgg_3d
            net = vgg_3d.__dict__[net_name](input_channels=self.channels,
                                            num_classes=self.num_classes)

        elif net_name.startswith('mvit'):
            from pytorchvideo.models.hub import vision_transformers as mvit
            net = mvit.__dict__[net_name](pretrained=False,
                                          input_channels=self.channels,
                                          head_num_classes=self.num_classes)
        elif net_name.startswith('x3d'):
            from pytorchvideo.models.hub import x3d as x3d
            net = x3d.__dict__[net_name](pretrained=False,
                                          input_channel=self.channels,
                                          model_num_class=self.num_classes)

        elif net_name.startswith('csn'):
            from pytorchvideo.models.hub import csn as csn
            net = csn.__dict__[net_name](pretrained=False,
                                          input_channel=self.channels,
                                          model_num_class=self.num_classes)

        if self.external_pretrained_path is not None:
            print('pretrain from: ', self.external_pretrained_path)
            model_dict = net.state_dict()
            ckpt = torch.load(self.external_pretrained_path,
                                map_location='cpu')
            pretrained_dict = {k: v for k, v in ckpt["model_state"].items() \
                if k in model_dict.keys() and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)

            net.load_state_dict(model_dict)
        if self.use_maxpool:
            print('Warning: use maxpooling for the classifer!')
            net.avgpool = nn.AdaptiveMaxPool3d((1, 1, 1))

        return net

    def _get_loss(self, loss_fun, class_weight=None):
        if class_weight is not None:
            class_weight = torch.tensor(class_weight)

        if loss_fun == 'Cross_Entropy':
            loss = nn.CrossEntropyLoss(class_weight)

        return loss

    def _get_optimizer(self, optimizer, net, lr):
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=lr,
                                         weight_decay=self.weight_decay)

        elif optimizer == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(),
                                        lr=lr,
                                        momentum=self.momentum)

        elif optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(net.parameters(),
                                          lr=lr,
                                          weight_decay=self.weight_decay)

        return optimizer

    def _get_lr_scheduler(self, lr_scheduler, optimizer):
        if lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, verbose=True)
        elif lr_scheduler == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.milestones, gamma=self.gamma)
        elif lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.T_max)
        elif lr_scheduler == 'CosineAnnealingWarmRestarts':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 20, T_mult=2)

        return lr_scheduler

    def _get_pre_trained(self, weight_path):
        checkpoint = torch.load(weight_path,map_location='cpu')
        print(f'>>>>load weight from: {weight_path}')
        try:
            self.net.load_state_dict(checkpoint['state_dict'])
            # self.start_epoch = checkpoint['epoch'] + 1
        except:
            model_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint["state_dict"].items() \
                if k in model_dict.keys() and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict)

# computing tools


class AverageMeter(object):
    '''
    Computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    '''
    Computes the precision@k for the specified values of k
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""
    def __init__(self,
                 patience=10,
                 verbose=True,
                 delta=0,
                 monitor='val_loss',
                 best_score=None,
                 op_type='min'):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type

        if self.op_type == 'min':
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):

        score = -val_score if self.op_type == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0

    def print_and_update(self, val_score):
        '''print_message when validation score decrease.'''
        if self.verbose:
            print(
                self.monitor,
                f'optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...'
            )
        self.val_score_min = val_score