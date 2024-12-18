import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import requests
import os
from tqdm import tqdm
import math

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer

from clip import clip

from clip.pmcclip import ModifiedResNet

from trainers.CoOp.coop_clip import load_clip_to_cpu
from trainers.prompt_templates import CUSTOM_TEMPLATES, BIOMEDCOOP_TEMPLATES

from transformers import AutoTokenizer, AutoModel

# Directory where the files should be located
directory = "clip/checkpoints"

# File URLs
pmcclip_files = {
    "text_encoder.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_encoder.pth",
    "image_encoder(resnet50).pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/image_encoder(resnet50).pth",
    "text_projection_layer.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_projection_layer.pth",
}

# File URLs
pubmedclip_files = {
    "PubMedCLIP_ViT32.pth": "https://huggingface.co/sarahESL/PubMedCLIP/resolve/main/PubMedCLIP_ViT32.pth?download=true",
}


# Function to download a file
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

@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME, device=self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits
    
@TRAINER_REGISTRY.register()
class ZeroshotPubMedCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # Check for files in the directory and download if necessary
        for filename, url in pubmedclip_files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")

        print(f"Loading PubMedCLIP (backbone: ViT-B/32)")
        clip_model, _ = clip.load("ViT-B/32", device=self.device)
        checkpoint = torch.load(os.path.join(directory,"PubMedCLIP_ViT32.pth"), map_location=self.device)
        clip_model.load_state_dict(checkpoint['state_dict'])

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits
    
@TRAINER_REGISTRY.register()
class ZeroshotBiomedCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        clip_model.eval().to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([tokenizer(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts,False)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model.eval()

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits
    
@TRAINER_REGISTRY.register()
class ZeroshotPMCCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # Check for files in the directory and download if necessary
        for filename, url in pmcclip_files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")


        print(f"Loading PMC-CLIP (backbone: RN50)")
        image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
        image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth')))
        text_encoder = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder.load_state_dict(torch.load(os.path.join(directory,'text_encoder.pth')))
        text_projection_layer = torch.load(os.path.join(directory,'text_projection_layer.pth'))
        text_projection_layer = nn.Parameter(text_projection_layer)
        self.text_encoder = text_encoder.to(self.device).eval()
        self.text_projection_layer = text_projection_layer.to(self.device)
        
        tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        # clip_model.eval().to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]

        tokenized_prompts = tokenizer(prompts, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        prompts = tokenized_prompts['input_ids'].to(self.device)

        with torch.no_grad():

            output = self.text_encoder(prompts.cuda(), attention_mask=tokenized_prompts['attention_mask'].cuda())
            pooler_output = output.pooler_output
            text_features = pooler_output @ self.text_projection_layer
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.image_encoder = image_encoder.to(self.device).eval()
        self.logit_scale = 4.4292
        

    def model_inference(self, image):
        image_features = self.image_encoder(image)
        if isinstance(image_features, dict):
            image_features = image_features['image_features']
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = math.exp(self.logit_scale)
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    templates = BIOMEDCOOP_TEMPLATES

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)


        num_temp = cfg.TRAINER.BIOMEDCOOP.N_PROMPTS
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i in range(num_temp):
            prompts = [BIOMEDCOOP_TEMPLATES[classname][i] for classname in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model

@TRAINER_REGISTRY.register()
class ZeroshotPubMedCLIP2(ZeroshotPubMedCLIP):
    """Prompt ensembling."""

    templates = BIOMEDCOOP_TEMPLATES

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # Check for files in the directory and download if necessary
        for filename, url in pubmedclip_files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")


        print(f"Loading PubMedCLIP (backbone: ViT-B/32)")
        clip_model, _ = clip.load("ViT-B/32", device=self.device)
        checkpoint = torch.load(os.path.join(directory,"PubMedCLIP_ViT32.pth"), map_location=self.device)
        clip_model.load_state_dict(checkpoint['state_dict'])

        for params in clip_model.parameters():
            params.requires_grad_(False)


        num_temp = cfg.TRAINER.BIOMEDCOOP.N_PROMPTS
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i in range(num_temp):
            prompts = [BIOMEDCOOP_TEMPLATES[classname][i] for classname in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model

@TRAINER_REGISTRY.register()
class ZeroshotBiomedCLIP2(ZeroshotBiomedCLIP):
    """Prompt ensembling."""

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        clip_model.eval().to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        num_temp = cfg.TRAINER.BIOMEDCOOP.N_PROMPTS
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i in range(num_temp):
            prompts = [BIOMEDCOOP_TEMPLATES[classname][i] for classname in classnames]
            prompts = torch.cat([tokenizer(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model

@TRAINER_REGISTRY.register()
class ZeroshotPMCCLIP2(ZeroshotPMCCLIP):
    """Prompt ensembling."""

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # Check for files in the directory and download if necessary
        for filename, url in pmcclip_files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")


        print(f"Loading PMC-CLIP (backbone: RN50)")
        image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
        image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth')))
        text_encoder = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder.load_state_dict(torch.load(os.path.join(directory,'text_encoder.pth')))
        text_projection_layer = torch.load(os.path.join(directory,'text_projection_layer.pth'))
        text_projection_layer = nn.Parameter(text_projection_layer)
        self.text_encoder = text_encoder.to(self.device).eval()
        self.text_projection_layer = text_projection_layer.to(self.device)
        self.image_encoder = image_encoder.to(self.device).eval()
        
        tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')

        for params in self.image_encoder.parameters():
            params.requires_grad_(False)

        for params in self.text_encoder.parameters():
            params.requires_grad_(False)

        num_temp = cfg.TRAINER.BIOMEDCOOP.N_PROMPTS
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i in range(num_temp):
            prompts = [BIOMEDCOOP_TEMPLATES[classname][i] for classname in classnames]
            tokenized_prompts = tokenizer(prompts, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
            prompts = tokenized_prompts['input_ids'].to(self.device)
            output = self.text_encoder(prompts.cuda(), attention_mask=tokenized_prompts['attention_mask'].cuda())
            pooler_output = output.pooler_output
            text_features = pooler_output @ self.text_projection_layer
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.logit_scale = 4.4292