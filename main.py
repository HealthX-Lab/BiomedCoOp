import os
import random
import argparse
import yaml
import time
import pandas as pd
from tqdm import tqdm
import requests

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

from clip.pmcclip import ModifiedResNet, image_transform

from transformers import AutoTokenizer, AutoModel

directory = "clip/checkpoints"

# File URLs
pubmedclip_files = {
    "PubMedCLIP_ViT32.pth": "https://huggingface.co/sarahESL/PubMedCLIP/resolve/main/PubMedCLIP_ViT32.pth?download=true",
}

# File URLs
pmcclip_files = {
    "text_encoder.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_encoder.pth",
    "image_encoder(resnet50).pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/image_encoder(resnet50).pth",
    "text_projection_layer.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_projection_layer.pth",
}

# Function to download a file with a progress bar
def download_file(url, filepath):
    print(f"Downloading {filepath}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, "wb") as file:
            # Use tqdm to show the progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    pbar.update(len(chunk))  # Update progress bar by the chunk size
        print(f"{filepath} downloaded successfully.")
    else:
        print(f"Failed to download {filepath}. HTTP Status Code: {response.status_code}")


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

class PMCCLIP(nn.Module):
    def __init__(self,image_encoder, text_encoder, projection_layer):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_projection_layer = projection_layer
        self.logit_scale = 4.4292
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
    def forward(self,image,text):
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        text_feature = self.text_encoder(input_ids)
        last_hidden_state = text_feature.last_hidden_state
        pooler_output = text_feature.pooler_output
        text_feature = pooler_output @ self.text_projection_layer
        image_feature = self.image_encoder(image)
        if isinstance(image_feature, dict):
            image_feature = image_feature['image_features']

        return image_feature, text_feature
    
    def encode_text(self, text):
        text_feature = self.text_encoder(text['input_ids'].cuda(), attention_mask=text['attention_mask'].cuda())
        pooler_output = text_feature.pooler_output
        text_feature = pooler_output @ self.text_projection_layer
        return text_feature
    
    def encode_image(self, image):
        image_feature = self.image_encoder(image)
        if isinstance(image_feature, dict):
            image_feature = image_feature['image_features']
        return image_feature

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

    elif(clip_model_pretrained == 'PubMedCLIP'):
        # Check for files in the directory and download if necessary
        for filename, url in pubmedclip_files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")
        clip_model, preprocess = clip.load('ViT-B/32')
        checkpoint = torch.load(os.path.join(directory,"PubMedCLIP_ViT32.pth"),weights_only=True)
        clip_model.load_state_dict(checkpoint['state_dict'])
        clip_model.eval()

    elif(clip_model_pretrained == 'PMCCLIP'):
        # Check for files in the directory and download if necessary
        for filename, url in pmcclip_files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")

        image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
        image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth'),weights_only=True))

        # Load Text Encoder
        text_encoder = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder.load_state_dict(torch.load(os.path.join(directory,'text_encoder.pth'),weights_only=True))

        # Load Text Proj Layer

        text_projection_layer = torch.load(os.path.join(directory,'text_projection_layer.pth'),weights_only=True)
        text_projection_layer = nn.Parameter(text_projection_layer)

        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image_encoder = image_encoder.to(device).eval()
        text_encoder = text_encoder.to(device).eval()
        text_projection_layer = text_projection_layer.to(device)

        clip_model = PMCCLIP(image_encoder, text_encoder, text_projection_layer).to(device).eval()
        preprocess = image_transform(image_size=224)

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
    print(f"Getting textual features as {clip_model_pretrained}'s classifier.")
    if(clip_model_pretrained in ['CLIP', 'PubMedCLIP']):
        clip_weights = clip_classifier(
            dataset.classnames, template, clip_model)
    elif(clip_model_pretrained == 'BiomedCLIP'):
        clip_weights = biomedclip_classifier(
            dataset.classnames, template, clip_model)
    elif(clip_model_pretrained == 'PMCCLIP'):
        clip_weights = pmcclip_classifier(
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

