import random
import torch
import numpy as np

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation f1 score decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class CutMix:
    def __init__(self, images, activate_cutmix):
        self.images = images # 현재 배치의 이미지들
        self.activate_cutmix = activate_cutmix # cutmix 수행 여부(list)
        self.lam = 1 #lamda
        self.rand_indexs = []

    def _rand_bbox(self, size, lam): # size : [B, C, W, H]
        W = size[2] # 이미지의 width
        H = size[3] # 이미지의 height
        cut_rat = np.sqrt(1. - lam)  # 패치 크기의 비율 정하기
        cut_w = np.int(W * cut_rat)  # 패치의 너비
        cut_h = np.int(H * cut_rat)  # 패치의 높이

        # uniform
        # 기존 이미지의 크기에서 랜덤하게 값을 가져옵니다.(중간 좌표 추출)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # 패치 부분에 대한 좌표값을 추출합니다.
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix(self, idx):
        lam = np.random.beta(1.0, 1.0)
        rand_index = random.choice(range(self.images.size()[0])) # 추출할 대상 index 한 개를 random 선택
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(self.images.size(), lam) #random한 bbox 좌표 추출
        self.images[idx, :, bbx1:bbx2, bby1:bby2] = self.images[rand_index, :, bbx1:bbx2, bby1:bby2]
        self.lam = 1 - ((bbx2-bbx1) * (bby2-bby1) / (self.images.size()[-1] * self.images.size()[-2])) # 1 - bbox의 넓이 / 이미지의 넓이 = bbox를 제외한 나머지 비중
        self.rand_indexs.append(rand_index) # bbox에 해당하는 이미지의 index 기록

    def __call__(self):
        for idx, _ in enumerate(self.images):
            # False면 cutmix 실행 X
            if not self.activate_cutmix[idx]:
                self.rand_indexs.append(idx)
                continue
            # cutmix=True일때
            self.cutmix(idx)
        return self.images, self.lam, self.rand_indexs

class CutMix_half:
    def __init__(self, images, activate_cutmix):
        self.images = images # 현재 배치의 이미지들
        self.activate_cutmix = activate_cutmix # cutmix 수행 여부(list)
        self.lam = 0.5 #lamda
        self.rand_indexs = []

    def _rand_bbox(self, size): # size : [B, C, W, H]
        W = size[2] # 이미지의 width

        return W // 2

    def cutmix(self, idx):
        rand_index = random.choice(range(self.images.size()[0])) # 추출할 대상 index 한 개를 random 선택
        half = self._rand_bbox(self.images.size()) #random한 bbox 좌표 추출
        self.images[idx, :, half // 2:, :] = self.images[rand_index, :, half // 2:, :]
        self.rand_indexs.append(rand_index) # bbox에 해당하는 이미지의 index 기록

    def __call__(self):
        for idx, _ in enumerate(self.images):
            # False면 cutmix 실행 X
            if not self.activate_cutmix[idx]:
                self.rand_indexs.append(idx)
                continue
            # cutmix=True일때
            self.cutmix(idx)
        return self.images, self.lam, self.rand_indexs
