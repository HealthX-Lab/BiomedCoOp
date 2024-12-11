import os
import argparse
import torch

from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer

biomedclip_model,_ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
biomedclip_model= biomedclip_model.float().eval()
parser = argparse.ArgumentParser()
parser.add_argument("--fpath", type=str, help="Path to the learned prompt")
parser.add_argument("--topk", default=1, type=int, help="Select top-k similar words")
args = parser.parse_args()


fpath = args.fpath
topk = args.topk

assert os.path.exists(fpath)

print(f"Return the top-{topk} matched words")

tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
token_embedding = biomedclip_model.text.transformer.embeddings.word_embeddings.weight

prompt_learner = torch.load(fpath, map_location="cpu")["state_dict"]
ctx = prompt_learner["prompt_learner.ctx"]
ctx = ctx.float()

if ctx.dim() == 2:
    # Generic context
    distance = torch.cdist(ctx, token_embedding)
    sorted_idxs = torch.argsort(distance, dim=1)
    sorted_idxs = sorted_idxs[:, :topk]
    token_embedding = token_embedding[sorted_idxs]
    output = ""
    for m, idxs in enumerate(sorted_idxs):
        words = [tokenizer.tokenizer.decode(idx.item()) for idx in idxs]
        dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
        output += f"{words[0]} ({dist[0]})\n"
    print(output) 

elif ctx.dim() == 3:
    # Class-specific context
    raise NotImplementedError
