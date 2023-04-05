from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch

#my_visible_devs = '0,1'  # '0, 3'  # 设置可运行GPU编号
#os.environ['CUDA_VISIBLE_DEVICES'] = my_visible_devs
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')

import json
import torch.utils.data
from torch.utils.data import random_split
from torchvision.transforms import transforms as T
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.train_factory import train_factory


def run(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task, opt.multi_scale)  # if opt.task==mot -> JointDataset

    f = open(opt.data_cfg)  # choose which dataset to train '../src/lib/cfg/mot15.json',
    data_config = json.load(f)
    trainset_paths = data_config['train']  # 训练集路径
    valset_paths   = data_config['test']
    dataset_root   = data_config['root']  # 数据集所在目录
    print("Dataset root: %s" % dataset_root)
    f.close()

    # Image data transformations
    #transforms_train = T.Compose([T.ToTensor()])
    #transforms_val   = T.Compose([T.ToTensor()])

    # Dataset train
    dataset_train    = Dataset(opt=opt,
                         root=dataset_root,
                         paths=trainset_paths,
                         img_size=opt.input_wh,
                         augment=True,
                         #transforms=transforms_train)
                         transforms=T.Compose([T.ToTensor()]))
    #opt = opts().update_dataset_info_and_set_heads(opt, dataset_train)
    #print("opt:\n", opt)

    # Dataset val
    dataset_val      = Dataset(opt=opt,
                         root=dataset_root,
                         paths=valset_paths,
                         img_size=opt.input_wh,
                         augment=False,
                         #transforms=transforms_val)
                         transforms=T.Compose([T.ToTensor()]))
    #opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    #print("opt:\n", opt)

    nID_dict_max = max(max(dataset_train.nID_dict.values()),max(dataset_val.nID_dict.values()))
    for k, v in dataset_train.nID_dict.items(): dataset_train.nID_dict[k] = nID_dict_max
    for k, v in dataset_val.nID_dict.items():   dataset_val.nID_dict[k]   = nID_dict_max


    opt = opts().update_dataset_info_and_set_heads(opt, dataset_train)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset_val)
    print("opt:\n", opt)    

    logger = Logger(opt)

    #os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str  # 多GPU训练
    #print("opt.gpus_str: ", opt.gpus_str)

    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')  # 设置GPU

    #opt.device = device
    #opt.gpus = my_visible_devs

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)

    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model,
                                                   opt.load_model,
                                                   optimizer,
                                                   opt.resume,
                                                   opt.lr,
                                                   opt.lr_step)

    # Get dataloader
    if opt.is_debug:
        if opt.multi_scale:
            train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       pin_memory=True,
                                                       drop_last=True)  # debug时不设置线程数(即默认为0)
            val_loader   = torch.utils.data.DataLoader(dataset=dataset_val,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       pin_memory=True,
                                                       drop_last=True)  # debug时不设置线程数(即默认为0)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       drop_last=True)  # debug时不设置线程数(即默认为0)
            val_loader   = torch.utils.data.DataLoader(dataset=dataset_val,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       pin_memory=True,
                                                       drop_last=True)  # debug时不设置线程数(即默认为0)
    else:
        if opt.multi_scale:
            train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       num_workers=opt.num_workers,
                                                       pin_memory=True,
                                                       drop_last=True)
            val_loader   = torch.utils.data.DataLoader(dataset=dataset_val,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       num_workers=opt.num_workers,
                                                       pin_memory=True,
                                                       drop_last=True)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       drop_last=True)  # debug时不设置线程数(即默认为0)
            val_loader   = torch.utils.data.DataLoader(dataset=dataset_val,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       pin_memory=True,
                                                       drop_last=True)  # debug时不设置线程数(即默认为0)


    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt=opt, model=model, optimizer=optimizer)
    # trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        
        # Train an epoch
        log_dict_train, _ = trainer.train(epoch, train_loader)

        logger.write('T_epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:  # mcmot_last_track or mcmot_last_det
            if opt.id_weight > 0:  # do tracking(detection and re-id)
                save_model(os.path.join(opt.save_dir, 'mcmot_last_track_' + opt.arch + '.pth'),
                           epoch, model, optimizer)
            else:  # only do detection
                # save_model(os.path.join(opt.save_dir, 'mcmot_last_det_' + opt.arch + '.pth'),
                #        epoch, model, optimizer)
                save_model(os.path.join(opt.save_dir, 'mcmot_last_det_' + opt.arch + '.pth'),
                           epoch, model, optimizer)
        logger.write('\n')

        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch % 3 == 0: # Val an epoch
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            
            # Val an epoch
            log_dict_val, _ =trainer.val(epoch, val_loader)
            logger.write('V_epoch: {} |'.format(epoch))
            for k, v in log_dict_val.items():
               logger.scalar_summary('val_{}'.format(k), v, epoch)
               logger.write('{} {:8f} | '.format(k, v))
            logger.write('\n')

    logger.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'  # '0, 1'
    opt = opts().parse()
    print("opt.gpus: ", opt.gpus)
    run(opt)
