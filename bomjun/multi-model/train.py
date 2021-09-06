import argparse
import glob
import json
import multiprocessing
import os
import random
import re
import copy
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.utils import class_weight
from tqdm import tqdm, tqdm_notebook
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MultiModelDataset
from loss import create_criterion
from util import EarlyStopping


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args, target_class, num_classes):
    seed_everything(args.seed)

    #save_dir = increment_path(os.path.join(model_dir, args.name))
    save_dir = os.path.join(model_dir, args.name)
    os.makedirs(save_dir, exist_ok=True)

    # -- settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MultiModelDataset
    dataset = dataset_module(
        data_dir=data_dir,
        target_label=target_class
    )

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: CustomAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: EfficientNetModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric      
    #criterion = create_criterion(args.criterion)  # default: weighted_cross_entropy
    target_label = dataset.get_labels(target_class)

    # a manual rescaling weight given to each class.
    weights = class_weight.compute_class_weight('balanced', np.unique(target_label), target_label)
    criterion= create_criterion(args.criterion)(weight=torch.tensor(weights).float().to(device))
    
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    #scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- early_stop
    early_stopping = EarlyStopping(patience=args.patience)
    
    # -- logging
#     logger = SummaryWriter(log_dir=save_dir)
#     with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
#         json.dump(vars(args), f, ensure_ascii=False, indent=4)

    state = {'best_val_loss': np.inf, 'best_val_acc': 0.}

    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(inputs)
            preds = torch.argmax(logits, dim=-1)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                #logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                #logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        #scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in tqdm(val_loader):
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = model(inputs)
                preds = torch.argmax(logits, dim=-1)

                loss_item = criterion(logits, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            
            early_stopping(val_loss, val_acc)
            if early_stopping.early_stop:
                print(f'Early Stopping!!!')
                break
                       
            if val_acc > state['best_val_acc'] and val_loss < state['best_val_loss']:
                print(f"New best model! val acc : {val_acc:4.2%}, val loss : {val_loss:4.2}! saving the best model..")
                state['model_state_dict'] = copy.deepcopy(model.state_dict())
                state['optimizer_state_dict'] = copy.deepcopy(optimizer.state_dict())
                state['val_loss'] = val_loss
                state['val_acc'] = val_acc          
                torch.save(state, f"{save_dir}/{target_class}_best.pt")
            
            state['best_val_loss'] = min(state['best_val_loss'], val_loss)
            state['best_val_acc'] = max(state['best_val_acc'], val_acc)
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {state['best_val_acc']:4.2%}, best loss: {state['best_val_loss']:4.2}"
            )
            #logger.add_scalar("Val/loss", val_loss, epoch)
            #logger.add_scalar("Val/accuracy", val_acc, epoch)
            print()
    
    del model, optimizer, criterion, early_stopping, train_loader, val_loader, dataset_module,dataset,\
        transform, transform_module, train_set, val_set, state
        
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)
    
    # 중요하지 않은 에러 무시
    import warnings
    warnings.filterwarnings(action='ignore')

    # 유니코드 깨짐현상 해결
    import matplotlib as mpl
    mpl.rcParams['axes.unicode_minus'] = False
    

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=2021, help='random seed (default: 2021)')
    parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train (default: 16)')
    parser.add_argument('--dataset', type=str, default='MultiModelDataset', help='dataset augmentation type (default: MultiModelDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: CustomAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training (default: [224, 224])')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--valid_batch_size', type=int, default=16, help='input batch size for validing (default: 16)')
    parser.add_argument('--model', type=str, default='EfficientNetModel', help='model type (default: EfficientNetModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='weighted_cross_entropy', help='criterion type (default: weighted_cross_entropy)')
    parser.add_argument('--patience', type=int, default=5, help='early stop patience (default:5)')
    #parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='multiModel', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/new_imgs'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    
    print(f'=================================== MASK MODEL TRAIN START ===================================')
    train(data_dir, model_dir, args, target_class='mask', num_classes=3)
    print(f'=================================== MASK MODEL TRAIN END ===================================')
    
    print(f'=================================== GENDER MODEL TRAIN START ===================================')
    train(data_dir, model_dir, args, target_class='gender', num_classes=2)
    print(f'=================================== GENDER MODEL TRAIN END ===================================')

    print(f'=================================== AGE MODEL TRAIN START ===================================')
    train(data_dir, model_dir, args, target_class='age', num_classes=3)
    print(f'=================================== AGE MODEL TRAIN END ===================================')