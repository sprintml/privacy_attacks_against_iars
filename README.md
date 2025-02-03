# Privacy Attacks on Image AutoRegressive Models

## Abstract

Image AutoRegressive generation has emerged as a new powerful paradigm. Specifically, image autoregressive models (IARs) surpass state-of-the-art diffusion models (DMs) in both image quality (FID: 1.48 vs. 1.58) and generation speed. However, the privacy risks associated with IARs remain unexplored, raising concerns regarding their responsible deployment. To address this gap, we conduct a comprehensive privacy analysis of IARs with respect to DMs, which serve as reference points. We develop a novel membership inference attack (MIA) that achieves an exceptionally high success rate in detecting training images (with a TPR@FPR=1\% of 86.38\% vs. 4.91\% for DMs). We leverage our novel MIA to provide dataset inference (DI) for IARs, which requires as few as 6 samples to detect dataset membership (compared to 200 for DI in DMs). Finally, we reconstruct hundreds of training data points from an IAR (e.g., 698 from VAR-d30). Our results demonstrate a fundamental privacy-utility trade-off: while IARs excel in image generation quality and speed, they are also significantly more vulnerable to privacy attacks compared to DMs. This trend suggests that utilizing techniques from DMs within IARs, such as modeling the per-token probability distribution using a diffusion procedure, can potentially help to mitigate the vulnerability of IARs to privacy attacks. 

## Requirements
A suitable conda environment named *iars_priv* can be created and activated with:

```
conda env create -f environment.yaml
conda activate iars_priv
git clone https://github.com/FoundationVision/VAR
git clone https://github.com/LTH14/mar
git clone https://github.com/bytedance/1d-tokenizer
mv 1d-tokenizer rar
```

Also, change `from models.diffloss import DiffLoss` to `from mar.models.diffloss import DiffLoss` in `mar/models/mar.py`.

In case of GBLICXX import error run `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:[YOUR_PATH_TO_CONDA]/envs/iars_priv/lib` (based on [this](https://stackoverflow.com/a/71167158))

## Downloading models

The scripts will download the models by themselves.

### Downloading data and data preparation

* ImageNet: Download [train](https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2) and [validation](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5) ImageNet LSVRC 2012 splits.

## Running Membership Inference

For each model, run
`python3 -u main.py +action=features_extraction +attack=$attack +model=$model +dataset=imagenet +dataset.split=$split`
Substitute $split with `train` and `val`. For attacking VAR and RAR set $attack=llm_mia_cfg, and for MAR: llm_mia_loss.

Then, run `analysis/mia_performance.py` to obtain TPR@FPR=1% for all IARs.

## Running Dataset Inference

Run `analysis/di.py`. Make sure to limit the cpu usage, e.g., by using `taskset -c 0-10 python3 analysis/di.py` to use only 10 cores, if you work in a shared computing ecosystem.

## Data Extraction

### Candidtates selection
```
for model in var_30 rar_xxl
do
    for idx in ${0..8}
    do
        python3 -u main.py +action=features_extraction +attack=mem_info +model=$model +dataset=imagenet +dataset.split=train cfg.run_id=1M_${idx} cfg.n_samples_eval=140000 dataset.gpu_cnt=8 dataset.gpu_idx=$idx
    done
done
for model in mar_h
do
    for idx in ${0..8}
    do
        python3 -u main.py +action=features_extraction +attack=mem_info_mar +model=$model +dataset=imagenet +dataset.split=train cfg.run_id=1M_${idx} cfg.n_samples_eval=140000 dataset.gpu_cnt=8 dataset.gpu_idx=$idx
    done
done
```
We suggest running these in a disributed GPU environment. The scripts are paralelizable.

### Generation

```
for model in var_30 rar_xxl mar_h
do
    python3 -u gen_memorized.py --model=$model --split=train
done
```

### Assesment

```
for model in var_30 rar_xxl mar_h
do
    python3 -u find_memorized.py --model=$model --split=train
done
```

Finally, a `${model}_memorized_train.csv` will be obtained. To find the memorized samples, do
```
df = pd.read_csv(f"{model}_memorized_train.csv")
print(df.loc[df.cosine_30>0.75].shape[0], "samples extracted from", model) # for VAR and RAR
print(df.loc[df.cosine_5>0.75].shape[0], "samples extracted from", model) # for MAR
```