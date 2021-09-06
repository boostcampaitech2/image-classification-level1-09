import numpy as np


#https://quokkas.tistory.com/entry/pytorch%EC%97%90%EC%84%9C-EarlyStop-%EC%9D%B4%EC%9A%A9%ED%95%98%EA%B8%B0
class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): epoch loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 epoch loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.epoch_loss_min = np.Inf
        self.delta = delta

#    def __call__(self, model, optimizer, epoch_loss, epoch_acc, epoch):
    def __call__(self, epoch_loss, epoch_acc):

        score = -epoch_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(model, optimizer, epoch_loss, epoch_acc, epoch)
            self.epoch_loss_min = epoch_loss ##########
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(model, optimizer, epoch_loss, epoch_acc, epoch)
            self.epoch_loss_min = epoch_loss #########
            self.counter = 0

#     def save_checkpoint(self, model, optimizer, epoch_loss, epoch_acc, epoch):
#         '''validation loss가 감소하면 모델을 저장한다.'''
#         if self.verbose:
#             print(f'Epoch loss decreased ({self.epoch_loss_min:.3f} --> {epoch_loss:.3f}).  Saving model ...')
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': epoch_loss
#         }, f'./saved/chkpoint_model_{epoch}_{epoch_loss:.3f}_{epoch_acc:.3f}.pt')
#         self.epoch_loss_min = epoch_loss
