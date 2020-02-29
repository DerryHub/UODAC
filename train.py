import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import TrainDataset, Resizer, Normalizer, Augmenter, collater
from src.efficientdet import EfficientDet
import numpy as np
from tqdm import tqdm
from config import get_args

def train(opt):
    num_gpus = 1
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        raise Exception('no GPU')

    cudnn.benchmark = True

    training_params = {"batch_size": opt.batch_size * num_gpus,
                       "shuffle": True,
                       "drop_last": False,
                       "collate_fn": collater,
                       "num_workers": opt.worker}

    training_set = TrainDataset(root_dir=opt.data_path, 
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    training_generator = DataLoader(training_set, **training_params)

    opt.num_classes = training_set.num_classes

    model = EfficientDet(opt)
    if opt.resume:
        print('Loading model...')
        model.load_state_dict(torch.load(os.path.join(opt.saved_path, opt.network+'.pth')))

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    model = model.cuda()
    model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    best_loss = np.inf
    model.train()

    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epochs):
        print('Epoch: {}/{}:'.format(epoch + 1, opt.num_epochs))
        model.train()
        epoch_loss = []
        progress_bar = tqdm(training_generator)
        for iter, data in enumerate(progress_bar):
            optimizer.zero_grad()
            cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])
            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()
            loss = cls_loss + reg_loss
            if loss == 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            epoch_loss.append(float(loss))
            total_loss = np.mean(epoch_loss)

            progress_bar.set_description('Epoch: {}/{}. Iteration: {}/{}'.format(epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch))
            
            progress_bar.write('Cls loss: {:.5f}\tReg loss: {:.5f}\tBatch loss: {:.5f}\tTotal loss: {:.5f}'.format(
                    cls_loss, reg_loss, loss, total_loss))

        loss = np.mean(epoch_loss)
        scheduler.step(loss)

        if loss + opt.es_min_delta < best_loss:
            print('Saving model...')
            best_loss = loss
            torch.save(model.module.state_dict(), os.path.join(opt.saved_path, opt.network+'.pth'))

if __name__ == "__main__":
    opt = get_args()
    train(opt)
