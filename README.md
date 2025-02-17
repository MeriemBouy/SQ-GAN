# SQ-GAN: Semantic Image Communication Using Masked Vector Quantization



### [📜 Paper](https://arxiv.org/abs/2502.09520)

**Authors:** [Francesco Pezone](https://scholar.google.com/citations?hl=it\&user=RAOXtOEAAAAJ), [Sergio Barbarossa](https://scholar.google.com/citations?hl=it\&user=2woHFu8AAAAJ), [Giuseppe Caire](https://scholar.google.com/citations?hl=it\&user=g66ErTcAAAAJ)

---

## 📌 Overview

SQ-GAN (Semantic Masked VQ-GAN) is a novel approach that leverages generative models for optimizing image compression in task-oriented communications. Our model selectively encodes semantically significant features using:

- **Semantic segmentation** for identifying key regions.
- **Semantic-Conditioned Adaptive Mask Module (SAMM)** for adaptive feature encoding.
- **Advanced semantic-based compression approach** outperforming JPEG2000 and BPG, particularly at extreme low bit rates.

### SQ-GAN framework
<img src='assets\scheme.png'> 

### Main Results
<img src='assets\results.png'> 

---

## 🚀 Installation & Setup
Run the following code
```bash
conda env create -f environment.yml
conda activate sqgan_env
```


---

## 📂 Dataset Preparation

SQ-GAN is trained on the **Cityscapes** dataset. To prepare the dataset:

1. Follow the dataset setup instructions from [SPADE](https://github.com/NVlabs/SPADE.git).
2. Update `DefaultDataPath.Cityscapes.root` in `data/default.py` with the correct dataset path.

---

## 🔧 Training SQ-GAN

SQ-GAN follows a **3-step training approach**, training individual subnetworks before final joint training.

### **Step 1: Train Semantic Subnetwork (G\_s)**

```bash
python3 train.py --mode ssm \
                 --gpus -1 \
                 --base config/sqgan_cityscapes.yml \
                 --max_epochs 150
```

### **Step 2: Train Image Subnetwork (G\_x)**

```bash
python3 train.py --mode img \
                 --gpus -1 \
                 --base config/sqgan_cityscapes.yml \
                 --max_epochs 150
```

### **Step 3: Train Final Model (G)**

Before final training, the model must be split into components to allow different parts to load simultaneously without issues:

#### **Splitting Checkpoints**

##### **Split G\_s**

```bash
python3 train.py --mode ssm \
                 --gpus -1 \
                 --base config/sqgan_cityscapes.yml \
                 --max_epochs 150 \
                 --resume_from_checkpoint /path/to/G_s_checkpoint.ckpt \
                 --split_path Final_ckpt_parts/
```

##### **Split G\_x**

```bash
python3 train.py --mode img \
                 --gpus -1 \
                 --base config/sqgan_cityscapes.yml \
                 --max_epochs 150 \
                 --resume_from_checkpoint /path/to/G_x_checkpoint.ckpt \
                 --split_path Final_ckpt_parts/
```

#### **Final Training**

```bash
python3 train.py --mode all \
                 --gpus -1 \
                 --base config/sqgan_cityscapes.yml \
                 --max_epochs 150 \
                 --ckpt_path_parts Final_ckpt_parts/
```

---

## 🎯 Pre-Trained Models

Pre-trained models for the **Cityscapes** dataset are available for download:

| Dataset    | Checkpoint Link                                                                                      |
| ---------- | ---------------------------------------------------------------------------------------------------- |
| Cityscapes | [📥 Download](https://drive.google.com/file/d/10N0pxmfLbm-D2lYTOX-Y0014YicF2u-q/view?usp=sharing) |

---

## 📊 Model Evaluation

### **Pre-requisites**

- **Batchsize**: in `config/sqgan_cityscapes.yml` set the dataset batchsize to 1
- **Semantic Segmentation Model**: SQ-GAN uses [InternImage](https://github.com/OpenGVLab/InternImage/tree/master) for generating segmentation maps.
- **Modify Paths**: In `sample.py`, update `internimage_path` with the correct InternImage path. If using another segmentation model, modify lines 185-186 accordingly.

### **Evaluation Methods**

**Option 1: Evaluate from split submodels**

```bash
python3 sample.py --mode all \
                  --base config/sqgan_cityscapes.yml \
                  --ckpt_path_parts Final_ckpt_parts/ \
                  --gpus -1
```

**Option 2: Evaluate from a single checkpoint**

```bash
python3 sample.py --mode all \
                  --base config/sqgan_cityscapes.yml \
                  --resume_from_checkpoint /path/to/merged_checkpoint.ckpt \
                  --gpus -1
```

Results are stored in the `Result/` folder. Multiple subfolders are crated for the desired combination of masking fractions and the performances are saved in `metrics.csv`.

---

## Acknowledgment
The code is based on [MQ-VAE](https://github.com/CrossmodalGroup/MaskedVectorQuantization)

---

## 🔗 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{SQ-GAN,
  title={SQ-GAN: Semantic Image Communication Using Masked Vector Quantization},
  author={Francesco Pezone, Sergio Barbarossa, Giuseppe Caire},
  journal={arXiv preprint arXiv:2502.09520},
  year={2025}
}
```


