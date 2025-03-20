from huggingface_hub import hf_hub_download 
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--task", type=str, default="few_shot")
parser.add_argument("--dataset", type=str, default="btmri")
parser.add_argument("--shots", type=int, default=1)
parser.add_argument("--trainer", type=str, default="BiomedCoOp_BiomedCLIP")
args = parser.parse_args()

if(args.task == "few_shot"):
    base_path = f"few_shot/{args.dataset}/shots_{args.shots}/{args.trainer}/nctx4_cscFalse_ctpend"
    model_name = "model.pth.tar-100"
else:
    base_path = f"base2new/train_base/{args.dataset}/shots_16/{args.trainer}/nctx4_cscFalse_ctpend"
    model_name = "model.pth.tar-50"

for seed in [1]:
    file_path = os.path.join(base_path, f"seed{seed}", "prompt_learner", model_name)
    download_path = hf_hub_download(repo_id="TahaKoleilat/BiomedCoOp",
                    filename=file_path,
                    local_dir=".",
                    repo_type="model")