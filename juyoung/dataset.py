import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import pandas as pd
from pandas_streaming.df import train_test_apart_stratify
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms as tr
from albumentations import  *
from albumentations.pytorch import *

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
    def __init__(self, mean, std, is_valid, age_labels=None, **args):
        self.is_valid = is_valid
        if age_labels:
            self.age_labels = age_labels
        self.curmix_target = [idx for idx, age_label in enumerate(self.age_labels) if age_label in [1, 4]]
        self.curmix_target = random.choices(self.curmix_target, k=int(len(self.curmix_target) * 0.5))

        if not self.is_valid:
            # train transform
            self.transform = Compose([
                # CenterCrop(224, 224),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
                # RandomBrightnessContrast(0.6, p=0.6),
                # GaussianBlur(blur_limit=(100, 100), p=0.6)
                
                # ToTensor(),
                # Normalize(mean=mean, std=std),
                # AddGaussianNoise()
            ])

        else:
            # valid transform
            self.transform = Compose([
                # CenterCrop(224, 224),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
                # RandomBrightnessContrast(0.6, p=0.6),
                # GaussianBlur(blur_limit=(100, 100), p=0.6)
                # Resize(resize, Image.BILINEAR),
                # ToTensor(),
                # Normalize(mean=mean, std=std),
                # AddGaussianNoise()
            ])

    def __call__(self, image, index=None):
        if not self.is_valid:
            # cutmix를 수행할 대상 이미지 index
            index = [i in self.curmix_target for i in index]
        else:
            index = [False] * len(index)
        
        image = np.array(image)
        images = [self.transform(image=im)['image'].unsqueeze(0) for im in image]

        return torch.cat(images, dim=0), index


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

# class AgeLabels_55(int, Enum):
#     """
#     55세 이상 -> 60세로 이동
#     """
#     YOUNG = 0
#     MIDDLE = 1
#     OLD = 2

#     @classmethod
#     def from_number(cls, value: str, count : int) -> int:
#         try:
#             value = int(value)
#         except Exception:
#             raise ValueError(f"Age value should be numeric, {value}")

#         total_count_55 = 4011 # 55세이상 60세미만인 사람 수

#         if value < 30:
#             return (cls.YOUNG, 0)
#         elif 55 <= value < 60 and count < int(total_count_55 * 0.5): # 지정한 비율만큼 60세로 이동
#             return (cls.OLD, 1)
#         elif value < 60:
#             return (cls.MIDDLE, 0)
#         else:
#             return (cls.OLD, 0)

class MaskBaseDataset(Dataset):

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
    groups = []
    all_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio # validation data ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles: # profile = directory
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile) # directory fill path
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)
                # age_label, count_bool = AgeLabels_55.from_number(age, self.count_55)
                # self.count_55 += count_bool # 55세 이상 60세미만이면 +1

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                print(image.shape)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    # def set_transform(self, transform):
    #     self.transform = transform

    def __getitem__(self, index):
        # assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        
        # image_transform, active_cutmix = self.transform(image, self.age_labels, index) # transfrom 결과
        transform = Compose([
                Resize(224, 224, Image.BILINEAR)
            ])
        
        return transform(image=image)['image'], multi_class_label, index

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return np.array(Image.open(image_path))

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
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        # self.count_55 = 0 # 55세이상 60세미만인 사람
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles) # 전체 길이
        n_val = int(length * val_ratio) # validation ratio만큼의 길이

        val_indices = set(random.choices(range(length), k=n_val)) # n_val만큼 random choice
        train_indices = set(range(length)) - val_indices # n_val만큼 choice된 것을 제외한 나머지
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir) # iamges폴더의 하위 폴더 목록
        profiles = [profile for profile in profiles if not profile.startswith(".")] # 숨은 폴더, 파일 제외
        split_profiles = self._split_profile(profiles, self.val_ratio) # train, valid data split

        cnt = 0
        for phase, indices in split_profiles.items(): # train : indice, val : indice
            for _idx in indices:
                profile = profiles[_idx] # current profile directory
                img_folder = os.path.join(self.data_dir, profile) # full path directory
                for file_name in os.listdir(img_folder): # current profile file list
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_") # directory name split
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)
                    # age_label, count_bool = AgeLabels_55.from_number(age, self.count_55)
                    # self.count_55 += count_bool # 55세 이상 60세미만이면 +1

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)
                    self.groups.append(id)
                    self.all_labels.append(self.encode_multi_class(mask_label, gender_label, age_label)) 
                    self.indices[phase].append(cnt)
                    cnt += 1
                    
        # print(len(list(filter(lambda x: x in list(range(2, 18, 3)), self.age_labels)))) # 60대 label 수 확인
        
    def split_dataset(self) -> List[Subset]:
        df = pd.DataFrame({"indexs":self.indices, "groups":self.groups, "labels":self.all_labels})

        train, valid = train_test_apart_stratify(df, group="groups", stratify="labels", test_size=self.val_ratio)
        train_index = train["indexs"].tolist()
        valid_index = valid["indexs"].tolist()
        return  [Subset(self, train_index), Subset(self, valid_index)]
        # return [Subset(self, indices) for phase, indices in self.indices.items()] # 지정한 index에 위치해 있는 data를 가져옴


class TestDataset(Dataset):
    def __init__(self, img_paths, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = Compose([
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        
    def __getitem__(self, index):
        image = np.array(Image.open(self.img_paths[index]))
        if self.transform:
            image = self.transform(image=image)['image']
        return image

    def __len__(self):
        return len(self.img_paths)
