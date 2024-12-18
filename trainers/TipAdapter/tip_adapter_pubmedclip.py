import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import numpy as np

from tqdm import tqdm
from trainers.method import FSCLIPmethod
from trainers.utils import build_cache_model, search_hp_tip, cls_acc


class TIPAdapter_PubMedCLIP(FSCLIPmethod):
    '''
    TIP Adapter and Tip-Adapter-F methods
    '''

    def __init__(self, args: argparse.Namespace):
        # self.normalize = args.normalize
        super().__init__(args)
        self.cfg = args
        self.lr = args['lr']
        self.epoch = args['train_epoch']
        self.shot = args['shots']
        self.init_beta = args['init_beta']
        self.init_alpha = args['init_alpha']
        self.finetune = args['finetune']

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

        cache_keys, cache_values = build_cache_model(self.cfg, model, train_loader)
        beta, alpha = self.cfg['init_beta'], self.cfg['init_alpha']
        
        # Feature Extraction for Validation
        print("\nExtracting visual features and labels from val set.")
        val_features, val_labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(val_loader)):
                images, target = images.cuda(), target.cuda()
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                val_features.append(image_features)
                val_labels.append(target)
        val_features, val_labels = torch.cat(val_features), torch.cat(val_labels)

        start_time = time.time()
        if not self.finetune:
            # Zero-shot CLIP
            clip_logits = 100. * val_features @ text_weights
            acc = cls_acc(clip_logits, val_labels)
            print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

            # Tip-Adapter
            
            affinity = val_features @ cache_keys
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            # cache_logits = beta * affinity @ cache_values
            
            tip_logits = clip_logits + cache_logits * alpha
            acc = cls_acc(tip_logits, val_labels)
            print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

            # Search Hyperparameters
            best_beta, best_alpha = search_hp_tip(self.cfg, cache_keys, cache_values, val_features, val_labels, text_weights)
            
            # Zero-shot CLIP
            clip_logits = 100. * test_features @ text_weights
            acc = cls_acc(clip_logits, test_labels)
            print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

            # Tip-Adapter    
            affinity = test_features @ cache_keys
            cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
            
            tip_logits = clip_logits + cache_logits * best_alpha
            acc = cls_acc(tip_logits, test_labels)
            print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))
            
            return None, acc
        
        # Enable the cached keys to be learnable
        adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(model.dtype).cuda()
        adapter.weight = nn.Parameter(cache_keys.t())
        
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=self.cfg['lr'], eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'] * len(train_loader))

        
        # alpha initialization
        if self.cfg["grid_search"]:
            best_acc = 0.0
            print("**** Searching for best initialization of alpha **** \n")
            for init_alpha in range(self.cfg['init_alpha_scale']):
                init_adapter = self.search_init_hp(init_alpha, beta, train_loader, model, cache_keys, cache_values, text_weights)
                affinity = init_adapter(val_features)
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * val_features @ text_weights
                tip_logits = clip_logits + cache_logits * init_alpha
                acc = cls_acc(tip_logits, val_labels)
                if acc > best_acc:
                    best_acc = acc
                    alpha = init_alpha
                    adapter = init_adapter
            print(alpha)
            print(beta)
        
        # Training Prodecure
        print("**** Start Training **** \n")
        best_acc, best_epoch = 0.0, 0
        for train_idx in range(self.cfg['train_epoch']):
            # Train
            adapter.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, self.epoch))

            for i, (images, target) in enumerate(tqdm(train_loader)):
                
                images, target = images.cuda(), target.cuda()
                print("Extraction")
                with torch.no_grad():
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                affinity = adapter(image_features)
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * image_features @ text_weights
                tip_logits = clip_logits + cache_logits * alpha

                loss = F.cross_entropy(tip_logits, target)

                acc = cls_acc(tip_logits, target)
                correct_samples += acc / 100 * len(tip_logits)
                all_samples += len(tip_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                

            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

            # Eval
            adapter.eval()

            affinity = adapter(val_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * val_features @ text_weights
            tip_logits = clip_logits + cache_logits * alpha
            acc = cls_acc(tip_logits, val_labels)

            print("**** Tip-Adapter-F's val accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = train_idx
                torch.save(adapter.weight, self.cfg['cache_dir'] + "/best_F_" + str(self.cfg['shots']) + "shots.pt")
        
        adapter.weight = torch.load(self.cfg['cache_dir'] + "/best_F_" + str(self.cfg['shots']) + "shots.pt", weights_only=True)
        print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

        """
        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ text_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)
        print("**** Tip-Adapter-F's test accuracy before search : {:.2f}. ****\n".format(acc))
        """
        print("Total time = {:.4f}".format(time.time()-start_time))
        # Search Hyperparameters
        best_beta, best_alpha = search_hp_tip(self.cfg, affinity, cache_values, val_features, val_labels, text_weights, adapter=adapter)
        print("\n-------- Evaluating on the test set. --------")
        
        affinity = adapter(test_features)
        cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ text_weights
        tip_logits = clip_logits + cache_logits * best_alpha
        acc = cls_acc(tip_logits, test_labels)
        print("**** Tip-Adapter-F's test accuracy after search: {:.2f}. ****\n".format(acc))
        
        return loss, acc

    def search_init_hp(self, alpha, beta, val_loader, model, cache_keys, cache_values, text_weights):
        adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(model.dtype).cuda()
        adapter.weight = nn.Parameter(cache_keys.t())
        
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=self.cfg['lr'], eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'] * len(val_loader))

        for val_idx in range(self.cfg['train_epoch']):
            # finetune on validation
            adapter.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Val Epoch: {:} / {:}'.format(val_idx, self.epoch))

            for i, (images, target) in enumerate(tqdm(val_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                affinity = adapter(image_features)
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * image_features @ text_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, target)
                loss = F.cross_entropy(tip_logits, target)

                correct_samples += acc / 100 * len(tip_logits)
                all_samples += len(tip_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

            # Eval
            adapter.eval()

        return adapter