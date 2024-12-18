import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import math
import os
import requests
from tqdm import tqdm
from collections import OrderedDict

from trainers.prompt_templates import CUSTOM_TEMPLATES

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy

import torch
import torch.nn.functional as F
from torch import nn

from clip.pmcclip import ModifiedResNet

from transformers import AutoTokenizer, AutoModel

# Directory where the files should be located
directory = "clip/checkpoints"

# File URLs
files = {
    "text_encoder.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_encoder.pth",
    "image_encoder(resnet50).pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/image_encoder(resnet50).pth",
    "text_projection_layer.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_projection_layer.pth",
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

class TextEncoder(nn.Module):
    def __init__(self, pmcclip_model):
        super().__init__()
        self.model = pmcclip_model
        self.dtype = torch.float32
        self.text_encoder = pmcclip_model.text_encoder
        self.text_projection_layer = pmcclip_model.text_projection_layer

    def forward(self, prompts,tokenized_prompts):

        output = self.text_encoder(inputs_embeds=prompts.cuda(), attention_mask=tokenized_prompts['attention_mask'].cuda())
        pooler_output = output.pooler_output
        text_feature = pooler_output @ self.text_projection_layer

        return text_feature

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, pmcclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.KGCOOP.N_CTX
        ctx_init = cfg.TRAINER.KGCOOP.CTX_INIT
        dtype = torch.float32
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        self.class_token_position = cfg.TRAINER.KGCOOP.CLASS_TOKEN_POSITION
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = self.tokenizer(ctx_init, padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids']
            with torch.no_grad():
                embedding = pmcclip_model.text_encoder.embeddings.word_embeddings(prompt.cuda()).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        bias_vectors = torch.empty(1, 512, dtype=dtype)
        nn.init.normal_(bias_vectors, std=0.02)
        self.bias_vectors = nn.Parameter(bias_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self.tokenizer(name, padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids']) \
                     for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
      
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts_}")

        with torch.no_grad():
            tokenized_prompts_ = self.tokenizer(prompts_, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
            x_tokenized = tokenized_prompts_['input_ids']
            x_tokenized_attn_masks = tokenized_prompts_['attention_mask']
            
            # Move tensors to GPU
            x_tokenized = x_tokenized.cuda()
            x_tokenized_attn_masks = x_tokenized_attn_masks.cuda()
            
            # Pass through the model
            text_features = pmcclip_model.text_encoder(x_tokenized, x_tokenized_attn_masks)
            pooler_output = text_features.pooler_output
            text_features = pooler_output @ pmcclip_model.text_projection_layer

        self.text_features = text_features

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(768, 512)),
            ("relu", nn.ReLU(inplace=True))
            #("linear2", nn.Linear(128, 512))
        ]))


        if cfg.TRAINER.KGCOOP.PREC == "fp16":
            self.meta_net.half()


        tokenized_prompts = self.tokenizer(prompts, padding='max_length', truncation=True, max_length=77, return_tensors='pt')

        with torch.no_grad():
            embedding = pmcclip_model.text_encoder.embeddings.word_embeddings(tokenized_prompts['input_ids'].cuda()).type(dtype)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS


        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.KGCOOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


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
        pooler_output = text_feature.pooler_output
        text_feature = pooler_output @ self.text_projection_layer
        image_feature = self.image_encoder(image)
        if isinstance(image_feature, dict):
            image_feature = image_feature['image_features']

        return image_feature, text_feature

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, pmcclip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, pmcclip_model)
        self.cfg = cfg
        self.ori_embedding = self.prompt_learner.text_features
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = pmcclip_model.image_encoder
        self.text_encoder = TextEncoder(pmcclip_model)
        self.logit_scale = pmcclip_model.logit_scale
        self.dtype = torch.float32
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        self.meta_net = self.prompt_learner.meta_net
        self.adapter = Adapter(768, 4).to(self.dtype)


    def forward(self, image):
        prompts = self.prompt_learner()
        image_features = self.image_encoder(image.type(self.dtype))
        if isinstance(image_features, dict):
            image_features = image_features['image_features']

        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts) 
        text_features_old = self.ori_embedding

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = math.exp(self.logit_scale)

        logits = logit_scale * image_features @ text_features.t()

        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        score = cos(text_features,text_features_old)
        score = 1.0-torch.mean(score)

        return logits, score


@TRAINER_REGISTRY.register()
class KgCoOp_PMCCLIP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.KGCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # Check for files in the directory and download if necessary
        for filename, url in files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")

        print(f"Loading PMC-CLIP (backbone: RN50)")
        image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
        image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth'), weights_only=True))

        # Load Text Encoder
        text_encoder = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder.load_state_dict(torch.load(os.path.join(directory,'text_encoder.pth'), weights_only=True))

        # Load Text Proj Layer

        text_projection_layer = torch.load(os.path.join(directory,'text_projection_layer.pth'), weights_only=True)
        text_projection_layer = nn.Parameter(text_projection_layer)

        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image_encoder = image_encoder.to(device).eval()
        text_encoder = text_encoder.to(device).eval()
        text_projection_layer = text_projection_layer.to(device)

        pmcclip_model = PMCCLIP(image_encoder, text_encoder, text_projection_layer).to(device).eval()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, pmcclip_model)
        self.w = cfg.TRAINER.KGCOOP.W

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            #if "prompt_learner" not in name: # and "adapter" not in name:
            if "ctx" not in name: 
                param.requires_grad_(False)
            else:
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        
        #self.optim_ = build_optimizer(self.model.adapter, cfg.OPTIM)
        #self.sched_ = build_lr_scheduler(self.optim, cfg.OPTIM)
        #self.register_model('clip_adapter', self.model.adapter, self.optim_, self.sched_)

        self.scaler = GradScaler() if cfg.TRAINER.KGCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.KGCOOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output,score = self.model(image)
            loss = F.cross_entropy(output, label)+self.w*score
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            #self.update_lr()
            self.sched.step()
            #self.sched_.step()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def model_inference(self, input):
        return self.model(input)[0]


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print(names)

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
