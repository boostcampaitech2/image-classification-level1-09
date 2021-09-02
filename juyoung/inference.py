import argparse
import os
from importlib import import_module

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
import pyramidnet as PYRM


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes,
        freeze = True
    )
    # model = PYRM.PyramidNet('imagenet', 32, 300, 18, True)
    model = torch.nn.DataParallel(model)
    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'checkpoint13.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'cropped_images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]

    # -- define dataset
    dataset1 = TestDataset(img_paths, trans_n=1)
    dataset2 = TestDataset(img_paths, trans_n=2)
    dataset3 = TestDataset(img_paths, trans_n=3)

    # -- define dataloader
    loader1 = torch.utils.data.DataLoader(
        dataset1,
        # batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    loader2 = torch.utils.data.DataLoader(
        dataset2,
        # batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    loader3 = torch.utils.data.DataLoader(
        dataset3,
        # batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")

    # -- save prediction
    all_preds = []
    all_logits = np.zeros((1, 18))
    pred_1 = []
    pred_2 = []
    pred_3 = []

    # -- inference(TTA)
    with torch.no_grad():
        for idx, images in enumerate(loader1):
            images = images.to(device)
            pred = model(images)
            all_logits = np.vstack([all_logits, pred.cpu()])
            # pred = pred.argmax(dim=-1)
            pred_1.extend((2*pred).cpu())
    with torch.no_grad():
        for idx, images in enumerate(loader2):
            images = images.to(device)
            pred = model(images)
            all_logits = np.vstack([all_logits, pred.cpu()])
            # pred = pred.argmax(dim=-1)
            pred_2.extend(pred.cpu())
    with torch.no_grad():
        for idx, images in enumerate(loader3):
            images = images.to(device)
            pred = model(images)
            all_logits = np.vstack([all_logits, pred.cpu()])
            # pred = pred.argmax(dim=-1)
            pred_3.extend((2*pred).cpu())

    for k in range(len(pred_1)):
        npred = torch.zeros(images.size(0), 18)
        npred = torch.add(npred, pred_1[k])
        npred = torch.add(npred, pred_2[k])
        npred = torch.add(npred, pred_3[k])
        npred = npred.argmax(dim=-1) 

        all_preds.extend(npred.cpu().numpy())

    info['ans'] = all_preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    np.save(os.path.join(output_dir, 'logit.npy'), all_logits)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for validing (default: 16)')
    parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='efficientnet_b7', help='model type (default: resnet50)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)