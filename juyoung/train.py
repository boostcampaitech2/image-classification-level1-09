import argparse
import glob
import os
import random
import re
from importlib import import_module
from pathlib import Path
from GPUtil import showUtilization as gpu_usage
from ipywidgets.widgets import widget
from utils import CutMix, CutMix_half

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import wandb

from loss import create_criterion

from tqdm.notebook import tqdm
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold

import pyramidnet as PYRM

def empty_cache():
    """
    GPU cache를 비우는 함수
    """
    print("Initial GPU Usage") 
    gpu_usage() 
    print("GPU Usage after emptying the cache") 
    torch.cuda.empty_cache() 
    gpu_usage()

def seed_everything(seed):
    """
    random seed를 고정하기위한 함수
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer):
    """
    optimizer에서 현재 learning rate를 가져오는 함수

    optimizer : 현재 사용하는 optimizer
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path) # model_dir(./model)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*") # 지정한 패턴에 맞는 파일을 불러옴
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}" # 뒷 부분에 숫자 + 1을 하여 return

def train(data_dir, model_dir, args):
    # gpu를 비우고 seed를 고정
    empty_cache()
    seed_everything(args.seed)

    # model을 저장할 path를 지정
    save_dir = increment_path(os.path.join(model_dir, args.name)) 

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    wandb.init(project="image_classification", entity='jujoo')

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir, # images directory
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    
    transform  = transform_module(
                            mean=dataset.mean,
                            std=dataset.std,
                            is_valid=False,
                            age_labels=dataset.age_labels
                        )

    transform_valid = transform_module(
        mean=dataset.mean,
        std=dataset.std,
        is_valid=True,
        age_labels=dataset.age_labels
    )

    # -- Out-Of-Fold Ensemble with TTA
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)

    counter = 0
    accumulation_steps = 2
    best_val_acc = 0
    best_val_loss = np.inf
    oof_pred = None

    labels = dataset.all_labels

    # for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, labels)):
    
    # < -- k-fold 시작 -- >

    # -- data_loader
    train_set, val_set = dataset.split_dataset() # OOF 적용시 train_index, valid_index 지정

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=args.num_worker,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )


    # --earlystopping
    # earlystop_module = getattr(import_module("utils"), args.earlystopping)
    # earlystopping = earlystop_module(patience=10, verbose=True, path=save_dir)

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes,
        freeze = False
    ).to(device)
    
    # model = PYRM.PyramidNet('imagenet', 32, 300, 18, True).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    # weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    # weights = torch.FloatTensor(weights).to(device)
    criterion = create_criterion(args.criterion, classes=num_classes)  # default: f1
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: AdamW
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.lr_decay_step
    )
    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    # -- logging
    # logger = SummaryWriter(log_dir=save_dir)
    # with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
    #     json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf

    for epoch in tqdm(range(args.epochs)):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        train_f1 = 0.
        for idx, train_batch in enumerate(train_loader):
            inputs, labels, index = train_batch

            inputs = inputs
            inputs, active_cutmixs = transform(inputs, index)
            inputs = inputs.to(device)
            inputs, lam, rand_indexs = CutMix_half(inputs, active_cutmixs)()
            labels = labels.to(device)

            # cutmix loss를 계산하기 위해 필요
            rand_target = torch.tensor([labels[idx] for idx in rand_indexs], device=device)
            
            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            # loss = criterion(outs, labels)

            # --cutmix loss
            # print("len labels", len(labels),labels)
            # print("len rand target", len(rand_target), rand_target)
            loss = criterion(outs, labels) * lam + criterion(outs, rand_target) * (1 - lam)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            train_f1 += f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')
            
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                train_f1 = train_f1 / args.log_interval # 현재 interval 동안의 train_f1 계산
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr} || "
                    f"training F1 Score {train_f1:4.4}"
                )

                wandb.log({'train/accuracy': train_acc, 'train/loss': train_loss, 'train/f1_score' : train_f1})

                # logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                # logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0
                train_f1 = 0.

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            # figure = None
            valid_f1, n_iter = 0., 0.
            for val_batch in val_loader:
                inputs, labels, index = val_batch
                inputs, _ = transform_valid(image=inputs, index=index)
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                valid_f1 += f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')
                n_iter += 1

                # if figure is None:
                #     inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                #     inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                #     figure = grid_image(
                #         inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                #     )
                
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            valid_f1 = valid_f1 / n_iter
            
            if val_acc > best_val_acc:
            #     print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
            #     torch.save(model.module.state_dict(), f"{save_dir}/best.pth") # best model 저장
                best_val_acc = val_acc
                
            # torch.save(model.module.state_dict(), f"{save_dir}/last.pth") # 마지막 모델 저장
            
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2} || "
                f"validation F1 Score {valid_f1:4.4}"
            )
            
            wandb.log({'train/accuracy': train_acc, 'train/loss': train_loss, 'train/f1_score' : train_f1,
                        'vaild/accuracy': val_acc, 'vaild/loss': val_loss, 'vaild/f1_score' : valid_f1})

            # earlystopping(valid_f1, model)
            # if earlystopping.early_stop:
            #     print("Early stopping")
            #     break

            # logger.add_scalar("Val/loss", val_loss, epoch)
            # logger.add_scalar("Val/accuracy", val_acc, epoch)
            # logger.add_figure("results", figure, epoch)
            # print()
        if epoch > 3:
            torch.save(model.state_dict(), os.path.join(model_dir, f"checkpoint{epoch}.pt")) # epoch 4이상부터 저장

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: CustomAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=32, help='inputcd .. batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=16, help='input batch size for validing (default: 16)')
    parser.add_argument('--earlystopping', type=str, default='EarlyStopping', help='EarlyStopping')
    parser.add_argument('--model', type=str, default='efficientnet_b7', help='model type (default: resnet50)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 1e-4)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='f1', help='criterion type (default: f1)')
    parser.add_argument('--lr_decay_step', type=int, default=1e-3, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--num_worker', type=int, default=4, help='num_worker')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/cropped_images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)