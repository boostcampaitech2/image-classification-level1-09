import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MultiModelDataset


def load_model(saved_model, target_class, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, f'{target_class}_best.pt')
    load_state = torch.load(model_path, map_location=device)
    model.load_state_dict(load_state['model_state_dict'])

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = {}
    for target_class, num_classes in zip(['mask', 'gender', 'age'], [3, 2, 3]):
        model[target_class] = load_model(model_dir, target_class, num_classes, device).to(device)
        model[target_class].eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    final_preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            
            m_logits = model['mask'](images)
            m_logits = m_logits.detach().cpu().numpy()

            g_logits = model['gender'](images)
            g_logits = g_logits.detach().cpu().numpy()

            a_logits = model['age'](images)
            a_logits = a_logits.detach().cpu().numpy()

            add_preds = []
            for idx in range(len(m_logits)):
                _temp = []
                for m in m_logits[idx]:
                    for g in g_logits[idx]:
                        for a in a_logits[idx]:
                            _temp.append(m+g+a)
                add_preds.append(_temp)

            final_preds.append(add_preds)
    
    final_preds = np.argmax(final_preds, 1)
                                 
    info['ans'] = final_preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training (default: [224, 224])')
    parser.add_argument('--model', type=str, default='EfficientNetModel', help='model type (default: EfficientNetModel)')
    parser.add_argument('--name', default='multiModel', help='model load from {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = os.path.join(args.model_dir, args.name)
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
