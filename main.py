import os
import random
import argparse
import yaml
import time
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from trainers import __dict__ as all_methods
from utils import *
from open_clip.src.open_clip import create_model_from_pretrained


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_config', default='configs/base.yaml',
        help='setting of Few-shot CLIP')
    parser.add_argument(
        '--dataset_config', default='configs/caltech101.yaml',
        help='dataset config')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.base_config)
    cfg.update(load_cfg_from_cfg_file(args.dataset_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg



def main():

    # Load config file
    cfg = get_arguments()

    cache_dir = os.path.join('./caches', cfg.DATASET.NAME)
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    method = all_methods[cfg['method']](args=cfg)

    clip_model_pretrained = cfg['clip_model']

    if(clip_model_pretrained == 'CLIP'):
        clip_model, preprocess = clip.load(cfg['backbone'])
        clip_model.eval()

    elif(clip_model_pretrained == 'BiomedCLIP'):

        # Load the model and config files from the Hugging Face Hub
        clip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        clip_model = clip_model.cuda()
        clip_model.eval()

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)

    cfg.DATASET.ROOT = cfg['root_path']
    cfg.SEED = 1
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATASET.NUM_SHOTS = cfg['shots']

    print("Preparing dataset.")
    dataset = build_dataset(cfg)
    classnames = dataset.classnames
    test_loader = build_data_loader(data_source=dataset.test, batch_size=100, is_train=False, tfm=preprocess, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    template = ['a photo of a {}.']

    # Textual features
    if(clip_model_pretrained == 'CLIP'):
        print("Getting textual features as CLIP's classifier.")
        clip_weights = clip_classifier(
            dataset.classnames, template, clip_model)
    elif(clip_model_pretrained == 'BiomedCLIP'):
        print("Getting textual features as BiomedCLIP's classifier.")
        clip_weights = biomedclip_classifier(
            dataset.classnames, template, clip_model)

    # Pre-load test features
    f_test_time = time.time()
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(
        cfg, "test", clip_model, test_loader)

    total_acc = 0
    predictions = []
    for i in range(cfg['tasks']):
        random.seed(i+1)
        torch.manual_seed(i+1)
        print("Start Training Task:{}".format(str(i+1)))
        few_shot_train_data = dataset.generate_fewshot_dataset_(cfg['shots'], split="train")
        few_shot_val_data = dataset.generate_fewshot_dataset_(cfg['shots'], split="val") 

        if cfg['finetune']:
            train_loader = build_data_loader(
                data_source=few_shot_train_data, batch_size=cfg["batch_size"], tfm=train_tranform, is_train=True, shuffle=True)
        else:
            train_loader = build_data_loader(
                data_source=few_shot_train_data, batch_size=cfg["batch_size"], tfm=train_tranform, is_train=True, shuffle=False)
        val_loader = build_data_loader(
            data_source=few_shot_val_data, batch_size=cfg["batch_size"], tfm=preprocess, is_train=False, shuffle=False)

        loss, acc = method(train_loader=train_loader,
                        val_loader=val_loader,
                        test_features=test_features,
                        test_labels=test_labels,
                        text_weights=clip_weights,
                        model=clip_model,
                        classnames=classnames)
        print('Final Accuracy on task {}: {}'.format(str(i+1), acc))
        predictions.append(acc)
    tasks_acc, tasks_std = compute_confidence_interval(predictions)
    test_stats = {}
    test_stats['acc'] = tasks_acc
    test_stats['std'] = tasks_std

    print('Total Accuracy and std on {} tasks: {:.4f} , {:.4f}'.format(
        str(cfg['tasks']), tasks_acc, tasks_std))
    if not os.path.exists(cfg['output_dir']):
        os.makedirs(cfg['output_dir'])
    csv_path = os.path.join(cfg['output_dir'], cfg.DATASET.NAME +".csv")
    write_to_csv(cfg, csv_path, test_stats)

def write_to_csv(cfg, path, test_stats):
    
    try:
        res = pd.read_csv(path)
    except:
        res = pd.DataFrame()
    records = res.to_dict('records')
    if cfg['method'] == "TIPAdapter" and cfg["finetune"]:
        test_stats['method'] = "TIPAdapter-F"
    else:
        test_stats['method'] = cfg['method']
    test_stats['acc'] = round(test_stats['acc'],4)
    test_stats['std'] = round(test_stats['std'],4)
    test_stats['num_shots'] = cfg['shots']
    test_stats['tasks'] = cfg['tasks']

    records.append(test_stats)
    # Save back to dataframe
    df = pd.DataFrame.from_records(records)
    df.to_csv(path, index=False)

if __name__ == '__main__':
    main()

