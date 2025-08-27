# GraphLoRA
This is an official implementation of KDD 25 paper GraphLoRA: Structure-Aware Contrastive Low-Rank Adaptation for Cross-Graph Transfer Learning.

## 📰 Update
 1. **2025-07**
    🛠️ We have released an updated version with detailed hyperparameter selection.
 2. **2024-12**
    🎉 Our paper *"GraphLoRA: Structure-Aware Contrastive Low-Rank Adaptation for Cross-Graph Transfer Learning"* has been accepted to the **KDD 2025**!
    📄 [Read the paper on arXiv](https://arxiv.org/abs/2409.16670)

## Requirements
```
python==3.11.5
torch==2.1.0
cuda==12.1
numpy==1.26.0
torch_geometric==2.4.0
```

## How to Run
You can easily run our code by

```
# Pre-training
python main.py --is_pretrain True

# Fine-tuning

# 콘솔 출력 (기본)
python main.py --pretrain_dataset PubMed --test_dataset CiteSeer --is_transfer True --seed 42

# 로그 파일로 저장
python main.py --pretrain_dataset PubMed --test_dataset CiteSeer --is_transfer True --seed 42 --use_logging True

# 커스텀 로그 디렉토리 사용
python main.py --pretrain_dataset PubMed --test_dataset CiteSeer --is_transfer True --seed 42 --use_logging True --log_dir ./experiments/logs
```
