import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pandas_streaming.df import train_test_apart_stratify
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *
from facenet_pytorch import MTCNN

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            #CenterCrop((224, 224)),
            Resize(resize, Image.BILINEAR),
            RandomHorizontalFlip(),
            RandomRotation([-26, 26]),
            ToTensor(),
            Normalize(mean=mean, std=std),
            #AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MultiModelDataset(Dataset):
    """
        3 Multi Model Dataset
        - Mask Dataset for 3 clsses.
        - Gender Dataset for 2 classes.
        - Age Dataset for 3 classes.
    """
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []
    multi_labels = []
    idxs = []
    groups = []
    
    def __init__(self, data_dir, target_label='multi', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), val_ratio=0.2):
        self.data_dir = data_dir
        self.target_label = target_label
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        cnt = 0
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                _id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label.value)
                self.gender_labels.append(gender_label.value)
                self.age_labels.append(age_label.value)
                self.multi_labels.append(self.encode_multi_class(mask_label.value, gender_label.value, age_label.value))
                self.idxs.append(cnt)
                self.groups.append(_id)
                cnt += 1

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        
        if self.target_label == 'multi':
            return image_transform, multi_label
        elif self.target_label == 'mask':
            return image_transform, mask_label
        elif self.target_label == 'gender':
            return image_transform, gender_label
        elif self.target_label == 'age':
            return image_transform, age_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]
    
    def get_labels(self, target):
        if target == 'multi':
            return self.multi_labels
        elif target == 'mask':
            return self.mask_labels
        elif target == 'gender':
            return self.gender_labels
        elif target == 'age':
            return self.age_labels

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 사람 별로 train 과 val 로 나눕니다. 또한 클래스 라벨 별로 균등하게 나누어 줍니다.
        pandas_streaming.df의 train_test_apart_stratify 함수를 이용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        """
        if self.target_label == 'multi':
            df = pd.DataFrame({"indexs":self.idxs, "groups":self.groups, "labels":self.multi_labels})
        elif self.target_label == 'mask':
            df = pd.DataFrame({"indexs":self.idxs, "groups":self.groups, "labels":self.mask_labels})
        elif self.target_label == 'gender':
            df = pd.DataFrame({"indexs":self.idxs, "groups":self.groups, "labels":self.gender_labels})
        elif self.target_label == 'age':
            df = pd.DataFrame({"indexs":self.idxs, "groups":self.groups, "labels":self.age_labels})

        train, valid = train_test_apart_stratify(df, group="groups", stratify="labels", test_size=self.val_ratio)
        train_index = train["indexs"].tolist()
        valid_index = valid["indexs"].tolist()

        return  [Subset(self, train_index), Subset(self, valid_index)]



class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), device='cpu'):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        self.device = device

    def __getitem__(self, index):
        image = self.face_detect(index)

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
    
    def face_detect(self, idx):
        mtcnn = MTCNN(keep_all=True, device=self.device)
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        #mtcnn 적용
        boxes,probs = mtcnn.detect(img)

        # boxes 확인
        if len(probs) > 1:
            pass 
            #print(boxes)
        if not isinstance(boxes, np.ndarray):
            #print('Nope!')
            # 직접 crop
            img = img[100:400, 50:350, :]
        else: # boexes size 확인
            xmin = int(boxes[0, 0])-30
            ymin = int(boxes[0, 1])-30
            xmax = int(boxes[0, 2])+30
            ymax = int(boxes[0, 3])+30

            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmax > 384: xmax = 384
            if ymax > 512: ymax = 512

            img = img[ymin:ymax, xmin:xmax, :]

        del mtcnn

        return Image.fromarray(img)
