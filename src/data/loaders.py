# torch.utils.data.Dataset stores the samples and their corresponding labels.
# torch.utils.data.DataLoader wraps an iterable around the Dataset.
import numbers
import os
import queue as Queue
import threading

import cv2
# import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.utils import sort_directories_by_file_count


class IdifFace(Dataset):
    def __init__(self, root_dir, local_rank, transform, num_classes):
        super(IdifFace, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.local_rank = local_rank
        self.imgidx, self.labels = self.scan(root_dir, num_classes)

    def scan(self, root, num_classes):
        imgidex = []
        labels = []
        lb = -1
        list_dir = os.listdir(root)
        list_dir.sort()

        for l in list_dir[:num_classes]:
            images = os.listdir(os.path.join(root,l))
            lb += 1
            for img in images:
                imgidex.append(os.path.join(l,img))
                labels.append(lb)

        return imgidex,labels

    def read_image(self,path):
        return cv2.imread(os.path.join(self.root_dir,path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img = self.read_image(path)
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __len__(self):
        return len(self.imgidx)


class CasiaWebFace(Dataset):
    def __init__(self, root_dir, local_rank, transform, num_classes, selective):
        super(CasiaWebFace, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.local_rank = local_rank
        self.imgidx, self.labels = self.scan(root_dir, num_classes, selective)

    def scan(self, root, num_classes, selective):
        imgidex = []
        labels = []
        lb = -1
        list_dir = os.listdir(root)
        list_dir.sort()

        if selective:
            sorted_directories = sort_directories_by_file_count(root)
            for l, file_count in sorted_directories[:num_classes]:
                images = os.listdir(os.path.join(root, l))
                lb += 1
                for img in images:
                    imgidex.append(os.path.join(l, img))
                    labels.append(lb)
        else:
            for l in list_dir[:num_classes]:
                images = os.listdir(os.path.join(root,l))
                lb += 1
                for img in images:
                    imgidex.append(os.path.join(l,img))
                    labels.append(lb)

        return imgidex,labels

    def read_image(self,path):
        return cv2.imread(os.path.join(self.root_dir,path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img = self.read_image(path)
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __len__(self):
        return len(self.imgidx)
'''
class MS1MV2(Dataset):
    def __init__(self, root_dir, local_rank, img_size, transform):
        super(MS1MV2, self).__init__()

        self.transform = transform
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)
'''
class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                if isinstance(self.batch[k], list):
                    self.batch[k] = [item.to(device=self.local_rank, non_blocking=True) if isinstance(item, torch.Tensor) else item for item in self.batch[k]]
                elif isinstance(self.batch[k], torch.Tensor):
                    self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


import albumentations
import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

PRE__MEAN = [0.485, 0.456, 0.406]
PRE__STD = [0.229, 0.224, 0.225]

import math
from typing import Iterator, Optional


class DistributedWeightedSampler(Sampler):
    def __init__(
        self,
        dataset_csv: str,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        replacement: bool = True,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        dataframe = pd.read_csv(dataset_csv)
        class_counts = dataframe.label.value_counts()
        self.sample_weights = [1 / class_counts[i] for i in dataframe.label.values]
        self.num_samples = len(dataframe)
        self.replacement = replacement
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        if not isinstance(self.replacement, bool):
            raise TypeError(
                "replacement should be a boolean value, but got replacement={}".format(
                    self.replacement
                )
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer value, but got num_samples={}".format(
                    self.num_samples
                )
            )

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        if self.drop_last and self.num_samples % self.num_replicas != 0:
            self.num_samples_per_replica = math.ceil(
                (self.num_samples - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples_per_replica = math.ceil(
                self.num_samples / self.num_replicas
            )

        self.total_size = self.num_samples_per_replica * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            torch.tensor(self.sample_weights),
            self.total_size,
            self.replacement,
            generator=g,
        ).tolist()

        if self.shuffle:
            indices = torch.tensor(indices)
            indices = indices[torch.randperm(len(indices), generator=g)].tolist()

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

def ApplyWeightedRandomSampler(dataset_csv):
    dataframe = pd.read_csv(dataset_csv) # head: image_path, label
    class_counts = dataframe.label.value_counts()

    sample_weights = [1/class_counts[i] for i in dataframe.label.values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataframe), replacement=True)
    return sampler

class TrainDataset(Dataset):

    def __init__(self, csv_file, input_shape=(224, 224)):
        # self.image_dir = image_dir
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = albumentations.Compose(
            [
                albumentations.Resize(height=256, width=256),
                albumentations.RandomCrop(height=input_shape[0], width=input_shape[0]),
                albumentations.HorizontalFlip(),
                albumentations.RandomGamma(gamma_limit=(80, 180)),  # 0.5, 1.5
                albumentations.RGBShift(
                    r_shift_limit=20, g_shift_limit=20, b_shift_limit=20
                ),
                albumentations.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.dataframe)

    def get_labels(self):
        return self.dataframe.iloc[:, 1]

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = cv2.imread(img_path)
        if image is None:
            raise Exception("Error: Image is None.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = 1 if label_str == "bonafide" else 0

        image = self.composed_transformations(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.float)

class TestDataset(Dataset):

    def __init__(self, csv_file, input_shape=(224, 224)):
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = albumentations.Compose(
            [
                albumentations.Resize(height=input_shape[0], width=input_shape[1]),
                albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = cv2.imread(img_path)
        if image is None:
            raise Exception("Error: Image is None.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = 1 if label_str == "bonafide" else 0

        image = self.composed_transformations(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.float), img_path
