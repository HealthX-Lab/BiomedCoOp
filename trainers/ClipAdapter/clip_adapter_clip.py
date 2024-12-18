import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import argparse
import numpy as np

from tqdm import tqdm
from trainers.method import FSCLIPmethod
from trainers.utils import cls_acc

from torch.optim.lr_scheduler import _LRScheduler

class ClipAdapter_CLIP(FSCLIPmethod):
    '''
    CLIP Adapter method
        @article{gao2021clip,
            title={CLIP-Adapter: Better Vision-Language Models with Feature Adapters},
            author={Gao, Peng and Geng, Shijie and Zhang, Renrui and Ma, Teli and Fang, Rongyao and Zhang, Yongfeng and Li, Hongsheng and Qiao, Yu},
            journal={arXiv preprint arXiv:2110.04544},
            year={2021}
        }
    
    '''

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.lr = args['lr']
        self.epoch = args['train_epoch']
        self.alpha = args['alpha_ca']
        self.cfg = args

    def forward(self,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_features: torch.tensor,
                test_labels: torch.tensor,
                text_weights: torch.tensor,
                model: nn.Module,
                classnames):
        """
        inputs:
            train_loader : torch.utils.data.DataLoader
            test_features : torch.Tensor of shape [test_data_size, 1024]
            test_labels : torch.Tensor of shape [test_data_size]
            text_weights : torch.Tensor of shape [num_shot*num_classes, 1024]
        """
        # model.float()
        
        cfg = self.cfg
        """
        if cfg["shots"] == 1:
            self.cfg['train_epoch'] = 50
        elif cfg["shots"] == 2 or cfg["shots"] == 4:
            self.cfg['train_epoch'] = 100
        else:
            self.cfg['train_epoch'] = 200
        """
        print(self.cfg['train_epoch'])
        print('Building custom CLIP')
        model.eval()
        clip_ad_model = CustomCLIP(model)
        clip_ad_model_val = copy.deepcopy(clip_ad_model)
        
        print('Turning off gradients in both the image and the text encoder')
        for name, param in clip_ad_model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)
                
        for name, param in clip_ad_model_val.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)
        
        clip_ad_model.cuda()
        clip_ad_model_val.cuda()
    

        # Feature Extraction for Validation
        print("\nExtracting visual features and labels from val set.")
        val_features, val_labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(val_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                val_features.append(image_features)
                val_labels.append(target)
        val_features, val_labels = torch.cat(val_features), torch.cat(val_labels)
        start_time = time.time() 
        # alpha initialization
        if cfg['search_alpha_ca']:
            best_acc = 0.0
            print("**** Searching for best initialization of alpha **** \n")
            alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            for init_alpha in alpha_list:
                clip_ad_model_val.adapter = self.search_init_hp(init_alpha, train_loader,clip_ad_model_val, model, text_weights)
                logits = clip_ad_model_val(val_features, text_weights, self.alpha)
                acc = cls_acc(logits, val_labels)
                print(init_alpha)
                print(acc)
                if acc > best_acc:
                    best_acc = acc
                    alpha = init_alpha
                    # adapter = init_adapter
                    
        else:
            alpha = cfg["alpha_ca"]
        print(alpha)

        #optimizer = torch.optim.SGD(clip_ad_model.adapter.parameters(), self.lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'] * len(train_loader), eta_min=1e-5)
        
        optimizer = torch.optim.SGD(clip_ad_model.adapter.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'])
        scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, self.cfg["WARMUP_EPOCH"],
                self.cfg["WARMUP_CONS_LR"]
            )
        
        # Train
        print('\nStart Training procedure')
           
        best_acc, best_epoch = 0.0, 0
        for train_idx in range(self.cfg['train_epoch']):
            # Train
            clip_ad_model.adapter.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, self.cfg['train_epoch']))

            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                logits = clip_ad_model(image_features, text_weights, alpha)

                loss = F.cross_entropy(logits, target)

                acc = cls_acc(logits, target)
                correct_samples += acc / 100 * len(logits)
                all_samples += len(logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            clip_ad_model.adapter.eval()
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))
            clip_ad_model.eval()
            logits = clip_ad_model(val_features, text_weights, self.alpha)
            acc = cls_acc(logits, val_labels)
            
            print("**** Clip-Adapter's val accuracy: {:.4f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = train_idx
                torch.save(clip_ad_model.adapter, self.cfg['cache_dir'] + "/best_clipA_" + str(self.cfg['shots']) + "shots.pt")
        # Evaluation
        print("Total time = {:.4f}".format(time.time()-start_time))
        # clip_ad_model.adapter = torch.load(self.cfg['cache_dir'] + "/best_clipA_" + str(self.cfg['shots']) + "shots.pt")
        
        print('\nStart evaluation on test set')
        clip_ad_model.eval()
        logits_test = clip_ad_model(test_features, text_weights, self.alpha) 

        acc_test = np.mean(logits_test.argmax(dim=1).cpu().numpy() ==  test_labels.cpu().numpy())*100.0

        return loss, acc_test

    def search_init_hp(self, alpha, val_loader, clip_ad_model, model, text_weights):
        optimizer = torch.optim.SGD(clip_ad_model.adapter.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'])
        scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, self.cfg["WARMUP_EPOCH"],
                self.cfg["WARMUP_CONS_LR"]
            )    
        # Train
        print('\nStart Training procedure')

        best_acc, best_epoch = 0.0, 0
        clip_ad_model.adapter.train()
        for train_idx in range(self.cfg['train_epoch']):
            # Train
            
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, self.cfg['train_epoch']))

            for i, (images, target) in enumerate(tqdm(val_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                logits = clip_ad_model(image_features, text_weights, self.alpha)

                loss = F.cross_entropy(logits, target)

                # acc = cls_acc(logits, target)
                # correct_samples += acc / 100 * len(logits)
                # all_samples += len(logits)
                # loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        # Eval
        clip_ad_model.adapter.eval()

        return clip_ad_model.adapter

    
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class CustomCLIP(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = Adapter(512, 4).to(clip_model.dtype)

            
    def forward(self, image_features, text_features, alpha):
        x = self.adapter(image_features)

        # alpha = 0.2
        image_features = alpha * x + (1 - alpha) * image_features
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features

        return logits
class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)
                

class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


