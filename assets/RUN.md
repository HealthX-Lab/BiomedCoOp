# Training and Evaluation

We provide bash scripts in [scripts/](../scripts) for each technique including prompt learning and other few-shot adaptation techniques.
Make sure to configure the dataset paths in environment variable `DATA` and run the commands from the main directory `BiomedCoOp/`.
Below we provide training and evaluation instructions for BiomedCoOp. The same instructions applies for all other techniques.


### Training time and compute
We train BiomedCoOp on each dataset with a batch size of 4 using a **single** NVIDIA A100 GPU. Currently, multi-GPU training is not supported. You will have to specify the GPU number that you want to use.

## BiomedCoOp

#### (1) Few-shot evaluation setting

The default training settings are provided in the config files at `configs/trainers/BiomedCoOp/few_shot`. All hyper-parameters can be modified using this config file.

Below, we provide instructions to train BiomedCoOp on any dataset. 

```bash
# All possible dataset values include [btmri, busi, chmnist, covid, ctkidney, dermamnist, kneexray, kvasir, lungcolon, octmnist, retina]

# CLIP Models include [CLIP, PubMedCLIP, PMCCLIP, BiomedCLIP]

# trains and evaluates in a few-shot setting on all 3 seeds
CUDA_VISIBLE_DEVICES=<GPU number> bash scripts/biomedcoop/few_shot.sh <data directory> <dataset> <nb of shots> <clip model to use>
# Example on BTMRI using 16 shots and the BiomedCLIP model on GPU 0
CUDA_VISIBLE_DEVICES=0 bash scripts/biomedcoop/few_shot.sh data btmri 16 BiomedCLIP
```

#### Averaging results over 3 seeds: 
Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– btmri/
|   |–– shots_16/
|   |   |–– BiomedCoOp_BiomedCLIP/
|   |   |   |–– nctx4_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
```

Now use the script `parse_test_res.py` and run the commands below to calculate the averaged results:
```bash
# prints averaged results
python parse_test_res.py output/btmri/shots_16/BiomedCoOp_BiomedCLIP/nctx4_cscFalse_ctpend --test-log
```

The above steps can be repeated for other individual datasets.

#### (2) Base-to-Novel class generalization setting

```bash
# All possible dataset values include [btmri, chmnist, covid, ctkidney, dermamnist, kneexray, kvasir, lungcolon, octmnist, retina]

# CLIP Models include [CLIP, PubMedCLIP, PMCCLIP, BiomedCLIP]

# trains and evaluates on base and novel classes
CUDA_VISIBLE_DEVICES=<GPU number> bash scripts/biomedcoop/base2new.sh <data directory> <dataset> <clip model to use>
# Example on BTMRI using the BiomedCLIP model on GPU 0
CUDA_VISIBLE_DEVICES=0 bash scripts/biomedcoop/base2new.sh data btmri BiomedCLIP
```

#### Averaging results over 3 seeds: 
Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– btmri/
|   |   |   |–– shots_16/
|   |   |   |   |–– BiomedCoOp_BiomedCLIP/
|   |   |   |   |   |–– nctx4_cscFalse_ctpend/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– btmri/
|   |   |   |–– shots_16/
|   |   |   |   |–– BiomedCoOp_BiomedCLIP/
|   |   |   |   |   |–– nctx4_cscFalse_ctpend/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Now use the script `parse_test_res.py` and run the commands below to calculate the averaged results:
```bash
# prints averaged results for base classes
python parse_test_res.py output/base2new/train_base/btmri/shots_16/BiomedCoOp_BiomedCLIP/nctx4_cscFalse_ctpend
# averaged results for novel classes
python parse_test_res.py output/base2new/test_new/btmri/shots_16/BiomedCoOp_BiomedCLIP/nctx4_cscFalse_ctpend --test-log
```

The above steps can be repeated for other individual datasets.

#### Reproducing Results

Our trained model checkpoints can be found on HuggingFace [here](https://huggingface.co/TahaKoleilat/BiomedCoOp)

Run the following scripts to use the checkpoints and get testing results. Note that the following scripts automatically download the desired model weights:

##### (1) Few-shot Evaluation

```bash
CUDA_VISIBLE_DEVICES=<GPU number> bash scripts/biomedcoop/eval_fewshot.sh <data directory> <dataset> <nb of shots>
# Example on BTMRI using 16 shots and the BiomedCLIP model on GPU 0
CUDA_VISIBLE_DEVICES=0 bash scripts/biomedcoop/eval_fewshot.sh data btmri 16
```

##### (2) Base-to-Novel Generalization

```bash
CUDA_VISIBLE_DEVICES=<GPU number> bash scripts/biomedcoop/eval_base2new.sh <data directory> <dataset> <nb of shots>
# Example on BTMRI using 16 shots and the BiomedCLIP model on GPU 0
CUDA_VISIBLE_DEVICES=0 bash scripts/biomedcoop/eval_base2new.sh data btmri 16
```

#### Training and Evaluating other techniques

For other techniques, we provide their corresponding configs and scripts as follows.

```
configs
|–– datasets/
|–– trainers/
|   |–– BiomedCoOp/
|   |–– CLIP_Adapter/
|   |–– CoCoOp/
|   |–– CoOp/
|   |–– KgCoOp/
|   |–– LP/
|   |–– LP2/
|   |–– ProGrad/
|   |–– TiP_Adapter/
|   |–– Zeroshot/
```

```
scripts
|–– biomedcoop/
|–– clip_adapter/
|–– cocoop/
|–– coop/
|–– kgcoop/
|–– linear_probe/
|–– linear_probe2/
|–– prograd/
|–– tip_adapter/
|–– zeroshot/
```

Please use the corresponding config and script files and follow the same instructions as provided for BiomedCoOp in order to train and evaluate the other variants. 

#### Acknowledgements
This file for running the methods has been borrowed from [MaPLe's](https://github.com/muzairkhattak/multimodal-prompt-learning/blob/main/docs/RUN.md) official repository.
