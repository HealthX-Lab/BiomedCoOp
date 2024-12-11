import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import time
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import argparse
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
from .method import FSCLIPmethod
from .utils import build_cache_model, search_hp_tip, cls_acc


class COOP(FSCLIPmethod):
    '''
    CoOp method
    '''

    def __init__(self, args: argparse.Namespace):

        super().__init__(args)
        self.cfg = args
        self.lr = args['lr']
        self.epoch = args['train_epoch']
        self.shot = args['shots']

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
        
        if self.cfg["PREC"] == "fp32" or self.cfg["PREC"] == "amp":
            # CLIP's default self.cfg["PREC"]ision is fp16
            model.float()
        model.eval()
        print("Building custom CLIP")
        coop_model = CustomCLIP(self.cfg, classnames, model).cuda()
        
        print("Turning off gradients in both the image and the text encoder")
        for name, param in coop_model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)       
        
 
        coop_model.to("cuda")
        # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(coop_model.prompt_learner, self.cfg.OPTIM)
        optimizer = torch.optim.SGD(coop_model.prompt_learner.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'])
        scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, self.cfg["WARMUP_EPOCH"],
                self.cfg["WARMUP_CONS_LR"]
            )

        self.scaler = GradScaler() if self.cfg["PREC"] == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            coop_model = nn.DataParallel(coop_model)
        
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
        
      
        # Training Prodecure
        print("**** Start Training **** \n")
        best_acc, best_epoch = 0.0, 0
        for train_idx in range(self.cfg['train_epoch']):
            # Train
            # coop_model.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, self.epoch))

            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()

                if self.cfg["PREC"] == "amp":
                    with autocast():
                        image_features = model.encode_image(images)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        output = coop_model(image_features)
                        loss = F.cross_entropy(output, target)
                    self.optim.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)                   
                    self.scaler.update()
                    
                else:
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    output = coop_model(image_features)
                    loss = F.cross_entropy(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                acc = cls_acc(output, target)
                correct_samples += acc / 100 * len(output)
                all_samples += len(output)
                loss_list.append(loss.item())
                

            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))
            
            
            # Eval
            coop_model.eval()

            output = coop_model(val_features)
            
            acc = cls_acc(output, val_labels)

            print("**** Coop's val accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = train_idx
                torch.save(coop_model.prompt_learner, self.cfg['cache_dir'] + "/best_F_" + str(self.cfg['shots']) + "shots.pt")
            
        coop_model.prompt_learner = torch.load(self.cfg['cache_dir'] + "/best_F_" + str(self.cfg['shots']) + "shots.pt")    
        coop_model.eval()
         
        output = coop_model(test_features)
        acc = cls_acc(output, test_labels)  

        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))
        
        print("**** COOP's test accuracy : {:.2f}. ****\n".format(acc))

        return loss, acc

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg["NCTX"]
        ctx_init = cfg["CTX_INIT"]
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            clip_model.to("cpu")
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg["CSC"]:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        clip_model.to("cpu")
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg["CLASS_TOKEN_POSITION"]

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
    
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image_features):
        # image_features = self.image_encoder(images.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

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