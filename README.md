# SOLAR

Official implementation draft for **SOLAR: Capacity-Aware Shared Adaptation of SAM2 for Optical-SAR Semantic Segmentation**.

SOLAR, short for **Shared-backbone Optical-SAR Low-rank Adaptation for Remote Sensing**, is a parameter-efficient multimodal semantic segmentation framework for optical-SAR remote sensing. It adapts a mostly frozen SAM2 Hiera-B+ image encoder with one shared LoRA update for both optical and SAR passes, while keeping sensor-specific processing lightweight and localized to input alignment.

## Highlights

- **Shared SAM2 adaptation**: optical and SAR images are encoded by the same pretrained SAM2 Hiera-B+ backbone and the same LoRA factors, avoiding duplicated foundation encoders.
- **Sensor-specific input alignment**: lightweight modality adapters map heterogeneous optical and SAR inputs into the three-channel representation expected by SAM2.
- **Cross-modal interaction and fusion**: bidirectional bottleneck cross-attention exchanges coarse semantic context, followed by multi-scale spatial gating at 1/4, 1/8, and 1/16 feature scales.
- **Detail residual refinement**: a disagreement-guided residual branch uses remaining optical-SAR feature differences to refine local structures and boundaries.
- **Optical-SAR evaluation**: experiments are conducted on WHU-OPT-SAR and PIE-RGB-SAR with multimodal, single-sensor, missing-modality, ablation, and parameter-efficiency analyses.

## Method Overview

SOLAR contains four main components:

1. **Modality adaptation module**  
   Optical and SAR inputs first pass through separate lightweight input adapters. The adapted observations are then evaluated through one shared SAM2 image encoder with LoRA inserted into the Q/K/V projections of the Hiera trunk. Most pretrained SAM2 parameters remain frozen.

2. **Cross-modal interaction module**  
   The 1/16 bottleneck features from optical and SAR streams interact through bidirectional cross-attention. The resulting multi-scale pyramids are fused with spatial gates that select sensor evidence at each location.

3. **Semantic decoder**  
   A multi-scale segmentation head combines the fused 1/16, 1/8, and 1/4 features and predicts full-resolution semantic logits, with an auxiliary output for deep supervision.

4. **Detail refinement module**  
   A residual branch compares fine-scale optical and SAR features and adds a small learned correction to the main logits, improving narrow structures, region interiors, and local boundaries.

The main model implementation is in [`solar.py`](./solar.py).

## Repository Structure

```text
SOLAR/
|-- configs/                         # Dataset and experiment configs
|-- sam2/                            # SAM2 modules
|-- sam2_configs/                    # SAM2 model configs
|-- semseg/                          # Datasets, losses, metrics, optimizers, schedulers
|-- solar.py                         # SOLAR model and SingleModalSAM2 baseline
|-- train.py                         # Training entry
`-- test.py                          # Checkpoint evaluation script
```

## Installation

```bash
conda create -n solar python=3.9 -y
conda activate solar

pip install -r requirements.txt
```

If the pinned PyTorch wheels do not match your CUDA version, install `torch` and `torchvision` following the official PyTorch instructions first, then install the remaining packages from `requirements.txt`.

## Data Preparation

The paper evaluates SOLAR on:

- **WHU-OPT-SAR**: optical and SAR land-cover segmentation with 8 classes: Background, Farmland, City, Village, Water, Forest, Road, and Others.
- **PIE-RGB-SAR**: RGB and SAR land-cover segmentation with 6 classes: Others, City, Road, Water, Forest, and Farmland.

Please download each dataset from its official source and follow its license. Then update the dataset root in the corresponding config file:

```yaml
DATASET:
  ROOT: /path/to/dataset
```

Main configs:

- [`configs/whuoptsar_detail_ohem_ema.yaml`](./configs/whuoptsar_detail_ohem_ema.yaml)
- [`configs/piergbsar_detail_ohem_ema.yaml`](./configs/piergbsar_detail_ohem_ema.yaml)

The dataset loaders expect the following layouts.

WHU-OPT-SAR pre-tiled layout:

```text
WHU-OPT-SAR/
|-- train/
|   |-- sar/
|   |-- opt/
|   `-- lbl/
|-- val/
|   |-- sar/
|   |-- opt/
|   `-- lbl/
`-- test/
    |-- sar/
    |-- opt/
    `-- lbl/
```

The WHU loader also supports raw large-tile mode:

```text
whu-opt-sar/
|-- sar/
|-- optical/
`-- lbl/
```

PIE-RGB-SAR layout:

```text
PIE-RGB-SAR/
|-- rgb/
|-- sar/
|-- label/
|-- train.txt
`-- val.txt
```

`train.txt` and `val.txt` should contain sample ids without suffix.

## Pretrained Weights

SOLAR uses SAM2 Hiera-B+ as the foundation backbone.

Download the official SAM2 Hiera-B+ checkpoint from the [SAM2 repository](https://github.com/facebookresearch/sam2). The original SAM2 release provides `sam2_hiera_base_plus.pt`.

Place the checkpoint as:

```text
checkpoints/sam2_hiera_base_plus.pt
```

Then update the checkpoint path in each config:

```yaml
MODEL:
  SAM2_CKPT: /path/to/sam2_hiera_base_plus.pt
```

The corresponding SAM2 config is included in this repository:

```text
sam2_configs/sam2_hiera_b+.yaml
```

## Trained Checkpoints

The trained SOLAR checkpoints for each dataset are available below.

| Dataset | Modalities | Checkpoint |
| --- | --- | --- |
| WHU-OPT-SAR | OPT + SAR | [WHUOptSAR_SAM2-Detail-OHEM-EMA_os](https://pan.baidu.com/s/1L_e1mfwJH06jyhoVZYTedw?pwd=m7j5) |
| PIE-RGB-SAR | RGB + SAR | [PIERGBSAR_SAM2-Detail-OHEM-EMA-PIE_os](https://pan.baidu.com/s/11qfRtkM09cgXZigUCxYq-Q?pwd=m7j5) |

## Training

Update dataset paths and SAM2 checkpoint paths in the config first, then run:

```bash
python train.py --cfg configs/whuoptsar_detail_ohem_ema.yaml
```

For PIE-RGB-SAR:

```bash
python train.py --cfg configs/piergbsar_detail_ohem_ema.yaml
```

The paper uses 1024 x 1024 inputs, 110 epochs, AdamW, Dice + OHEM loss, LoRA rank 16, LoRA scale 16, auxiliary weight 0.4, EMA decay 0.999, and an effective batch size of 32 through gradient accumulation.

## Evaluation

Evaluate a trained checkpoint with:

```bash
python test.py --cfg configs/whuoptsar_detail_ohem_ema.yaml --model-path /path/to/checkpoint.pth
```

For PIE-RGB-SAR:

```bash
python test.py --cfg configs/piergbsar_detail_ohem_ema.yaml --model-path /path/to/checkpoint.pth
```

By default, `test.py` evaluates the validation split. You can select another split with:

```bash
python test.py --cfg configs/whuoptsar_detail_ohem_ema.yaml --model-path /path/to/checkpoint.pth --split test
```

## Main Results

### WHU-OPT-SAR

| Method | Input sensors | Backbone | OA | Kappa | mIoU |
| --- | --- | --- | ---: | ---: | ---: |
| DeepLabv3+ | Optical-SAR | - | 79.79 | 71.28 | 44.30 |
| PSCNN | Optical-SAR | - | 81.24 | 73.29 | 48.51 |
| MCANet | Optical-SAR | - | 81.67 | 74.01 | 50.80 |
| ASANet | Optical-SAR | - | 83.31 | 76.38 | 55.14 |
| CLFSam | Optical-SAR | SAM2 Hiera-B+ | 81.40 | 73.65 | 50.91 |
| SOLAR | Optical-SAR | SAM2 Hiera-B+ | 84.59 | 78.25 | 61.70 |

### PIE-RGB-SAR

| Method | Input sensors | Backbone | OA | Kappa | mIoU |
| --- | --- | --- | ---: | ---: | ---: |
| DeepLabv3+ | RGB-SAR | - | 86.31 | 80.42 | 69.57 |
| PSCNN | RGB-SAR | - | 86.58 | 80.85 | 71.79 |
| MCANet | RGB-SAR | - | 85.46 | 79.30 | 69.51 |
| ASANet | RGB-SAR | - | 89.43 | 85.00 | 77.75 |
| CLFSam | RGB-SAR | SAM2 Hiera-B+ | 86.76 | 81.19 | 72.24 |
| SOLAR | RGB-SAR | SAM2 Hiera-B+ | 90.38 | 86.36 | 79.69 |

## Ablation

On WHU-OPT-SAR:

| MAM input adapter | CMI cross-attention | CMI fusion | DRM | OA | Kappa | mIoU |
| --- | --- | --- | --- | ---: | ---: | ---: |
| No | Yes | Spatial gating | Yes | 84.34 | 77.89 | 59.64 |
| Yes | No | Spatial gating | Yes | 84.51 | 78.13 | 61.20 |
| Yes | Yes | Spatial gating | No | 84.53 | 78.17 | 60.50 |
| Yes | Yes | Averaging | Yes | 84.47 | 78.09 | 61.23 |
| Yes | Yes | Concatenation | Yes | 84.52 | 78.14 | 61.18 |
| Yes | Yes | Scalar weighting | Yes | 84.51 | 78.15 | 61.01 |
| Yes | Yes | Spatial gating | Yes | 84.59 | 78.25 | 61.70 |

## Missing-Modality Evaluation

On WHU-OPT-SAR, models are trained with both sensors and evaluated without retraining after one raw modality is replaced by zeros.

| Method | SAR-only OA | SAR-only Kappa | SAR-only mIoU | Optical-only OA | Optical-only Kappa | Optical-only mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepLabv3+ | 44.44 | 28.56 | 13.51 | 48.69 | 20.43 | 11.17 |
| PSCNN | 47.71 | 31.59 | 15.82 | 41.66 | 30.45 | 19.59 |
| MCANet | 31.05 | 16.15 | 9.17 | 43.33 | 27.64 | 13.78 |
| ASANet | 34.52 | 19.56 | 20.71 | 53.26 | 38.13 | 20.31 |
| SOLAR | 65.84 | 50.29 | 27.99 | 72.27 | 62.44 | 50.42 |


## License

This code is released for academic research purposes. Please follow the licenses and terms of the third-party code, datasets, and pretrained checkpoints used by this project.

