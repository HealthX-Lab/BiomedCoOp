import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os

from tqdm import tqdm
from trainers.method import FSCLIPmethod


class LinearProbe_PMCCLIP(FSCLIPmethod):
    '''
    Linear Probe method
    '''

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.lr = args['lr']
        self.epoch = args['train_epoch']
        self.shot = args['shots']
        self.num_step = args['num_step']

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

        y_loss = {}  # loss history
        y_loss['train'] = []

        x_epoch_loss = []
        fig_loss = plt.figure()
        ax0 = fig_loss.add_subplot(121, title="loss")


        def draw_curve_loss(current_epoch):
            x_epoch_loss.append(current_epoch)
            ax0.plot(x_epoch_loss, y_loss['train'], 'bo-', label='train')
            if current_epoch == 0:
                ax0.legend()
            fig_loss.savefig(os.path.join('./','LP_randominit'+'_'+str(self.shot)+'shots_loss.png'))


        # Feature Extraction for Training
        print("\nExtracting visual features and labels from train set.")
        features, labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)  
        features, labels = torch.cat(features), torch.cat(labels)
        
        
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
        

        # Perform logistic regression
        # classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
        # classifier.fit(features.cpu().numpy(), labels.cpu().numpy())

        # # Evaluate using the logistic regression classifier
        # predictions = classifier.predict(test_features.cpu().numpy())
        # acc_test = np.mean((test_labels.cpu().numpy() == predictions).astype(float)) * 100.

        start = time.time()
        # search initialization
        search_list = [1e6, 1e4, 1e2, 1, 1e-2, 1e-4, 1e-6]
        acc_list = []
        for c_weight in search_list:
            clf = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_weight).fit(features.cpu().numpy(), labels.cpu().numpy())
            pred = clf.predict(val_features.cpu().numpy())
            acc_val = sum(pred == val_labels.cpu().numpy()) / len(val_labels.cpu().numpy())
            acc_list.append(acc_val)

        # print(acc_list, flush=True)

        # binary search
        peak_idx = np.argmax(acc_list)
        c_peak = search_list[peak_idx]
        c_left, c_right = 1e-1 * c_peak, 1e1 * c_peak

        test_acc_step_list = np.zeros([self.num_step])

        def binary_search(c_left, c_right, step, test_acc_step_list):
            clf_left = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_left).fit(features.cpu().numpy(), labels.cpu().numpy())
            pred_left = clf_left.predict(val_features.cpu().numpy())
            acc_left = sum(pred_left == val_labels.cpu().numpy()) / len(val_labels.cpu().numpy())
            print("Val accuracy (Left): {:.2f}".format(100 * acc_left), flush=True)

            clf_right = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_right).fit(features.cpu().numpy(), labels.cpu().numpy())
            pred_right = clf_right.predict(val_features.cpu().numpy())
            acc_right = sum(pred_right == val_labels.cpu().numpy()) / len(val_labels.cpu().numpy())
            print("Val accuracy (Right): {:.2f}".format(100 * acc_right), flush=True)

            # find maximum and update ranges
            if acc_left < acc_right:
                # c_final = c_right
                clf_final = clf_right
                # range for the next step
                c_left = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_right = np.log10(c_right)
            else:
                # c_final = c_left
                clf_final = clf_left
                # range for the next step
                c_right = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_left = np.log10(c_left)

            pred = clf_final.predict(test_features.cpu().numpy())
            test_acc = 100 * sum(pred == test_labels.cpu().numpy()) / len(pred)
            print("Test Accuracy: {:.2f}".format(test_acc), flush=True)
            test_acc_step_list[step] = test_acc
            return np.power(10, c_left), np.power(10, c_right), step, test_acc_step_list

        for step in range(self.num_step):
            print('step is ',step)
            c_left, c_right, step, test_acc_step_list = binary_search(c_left, c_right, step, test_acc_step_list)
        # save results of last step
        acc_test_final = test_acc_step_list[-1]
        end = time.time()
        total_time = end-start
        print('total time is ',total_time)
        return 0, acc_test_final
