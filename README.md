# Image Classification
- Pstage Image classification(마스크 착용 여부 문류)의 최종 제출 코드

## Hardware
- GPU : V100
- Language : Python
- Develop tools : Jupyter Notebook, VSCode

## Prepare Images
- 다음과 같은 구성으로 Data 파일 추가하여 학습
```
data
  ㄴ train
      ㄴ images
      ㄴ train.csv
  ㄴ eval
      ㄴ images
      ㄴ info.csv
```

## Getting Started
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
```
pip install -r requirements.txt
```

### Training
```
SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py
```
### Inference
- TTA(Test Time Augmentation)을 사용한 infernece
```
SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py
```

## Apply
### Dataset
- MaskBaseDataset : 이미지를 기준으로 나눈 Dataset

### Model
- swin_large_patch4_window7_224

### Optimizer & Loss
- Optimizer : AdamW
- Loss : Weighted CrossEntropyLoss

### lr_scheduler
- CosineAnnealingLR

### Stratified K-Fold
