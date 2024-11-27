import os

os.environ['HF_HUB_CACHE'] = 'huggingface'
import itertools
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.model_selection import train_test_split

import lightning as L
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.utilities import grad_norm

from utils import TwoCropTransform, TwoCropTransformAlbumentation
from losses import SupConLoss

SEED = 42
seed_everything(42)


def read_dataset(dataset_root, dataset_name, class_rename=None):
    dataset = {}
    for folder in os.listdir(dataset_root):
        current_path = os.path.join(dataset_root, folder)
        if os.path.isdir(current_path):
            csv_file = os.path.join(current_path, "_annotations.csv")
            df = pd.read_csv(csv_file)
            df = df[df['filename'].map(df['filename'].value_counts()) == 1]
            df = df.reset_index(drop=True)
            df['filename'] = df['filename'].apply(lambda x: os.path.join(current_path, x))
            if class_rename is not None:
                df['class'] = df['class'].apply(lambda x: class_rename.get(x, x))
            df['class'] = df['class'].apply(lambda x: f"{dataset_name}__{x}")
            if folder == 'train':
                classes = df['class'].unique().tolist()
                dataset['labels'] = classes
            # df['class'] = df['class'].apply(lambda x: class2idx[x])
            dataset[folder] = df

    if "test" not in dataset and "valid" not in dataset:
        train_df, val_df = train_test_split(dataset['train'], test_size=0.2, random_state=SEED)
        dataset['train'] = train_df.reset_index(drop=True)
        dataset['test'] = val_df.reset_index(drop=True)

    return dataset


class CustomDataset(Dataset):
    def __init__(self, df, classes, transform=None):
        self.df = df.reset_index(drop=True)
        self.class2idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __getitem__(self, index):
        # Image
        image = cv2.imread(self.df.iloc[index].filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        # Label
        label = self.df.iloc[index]['class']
        return image, self.class2idx[label]

    def __len__(self):
        return len(self.df)


class CustomDataModule(L.LightningDataModule):
    def __init__(self, df_train, df_val, classes, train_transform=None, val_transform=None, batch_size=32):
        super().__init__()
        self.df_train = df_train
        self.df_val = df_val
        self.classes = classes
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = CustomDataset(
            df=self.df_train,
            classes=self.classes,
            transform=self.train_transform,
        )
        self.val_dataset = CustomDataset(
            df=self.df_val,
            classes=self.classes,
            transform=self.val_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
        )


class ViTSupCon(L.LightningModule):
    def __init__(self, model_name, embedding_dim=128, learning_rate=5e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, pretrained=True)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.model.head = nn.Linear(self.model.head.in_features, embedding_dim)
        self.learning_rate = learning_rate
        self.criterion = SupConLoss()

    def forward(self, images):
        return self.model(
            images
        )

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = torch.cat([images[0], images[1]], dim=0)
        bsz = labels.shape[0]
        features = self.model(images)
        features = F.normalize(features, dim=1)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.criterion(features, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=8)  # T_max is the number of epochs for a full cycle
        return [optimizer], [scheduler]


# model = timm.create_model("hf_hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
# print(model)

if __name__ == '__main__':
    usd_class_rename = {
        'fifty-back': '50',
        'fifty-front': '50',
        'twenty-back': '20',
        'twenty-front': '20',
        'ten-back': '10',
        'ten-front': '10',
        'five-back': '5',
        'five-front': '5',
        'one-back': '1',
        'one-front': '1'
    }

    datasets_list = [
        ("datasets/VND-1", "VND", None),
        ("datasets/Malaysian-Banknote-3", "MYR", None),
        ("datasets/money-detection-1", "SGD", None),
        ("datasets/euro_large-1", "EUR", None),
        ("datasets/Dollar-Bill-Detection-22", "USD", usd_class_rename),
        ("datasets/GBP", "GBP", None),
        ("datasets/AUD", "AUD", None),
        ("datasets/CNY", "CNY", None),
    ]

    datasets = []
    for d in datasets_list:
        datasets.append(read_dataset(*d))
    train_df = pd.concat([dataset['train'] for dataset in datasets], ignore_index=True)
    valid_df = pd.concat([dataset.get('valid', dataset['test']) for dataset in datasets], ignore_index=True)
    classes = list(itertools.chain(*(dataset['labels'] for dataset in datasets)))
    train_df = train_df[train_df.width == train_df.height]
    valid_df = valid_df[valid_df.width == valid_df.height]
    # Data transformation
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    mean = [0.50832066, 0.47808658, 0.43113454]
    std = [0.3145969, 0.30568244, 0.30669917]
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ], p=0.8),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.ToGray(p=0.2),
        A.Resize(224, 224),
        A.Normalize(mean=mean, std=std),  # Replace with your mean/std
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=mean, std=std),  # Replace with your mean/std
        ToTensorV2()
    ])

    MODEL_NAME = "hf_hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k"
    BATCH_SIZE = 128
    LEARNING_RATE = 5e-5

    data_module = CustomDataModule(train_df, valid_df, classes, TwoCropTransformAlbumentation(train_transform),
                                   val_transform, batch_size=BATCH_SIZE)
    data_module.setup()
    checkpoint_path = "./currency-recognition/z4exxl19/checkpoints/epoch=29-step=3360.ckpt"
    checkpoint_path = "./currency-recognition/zh732ypf/checkpoints/epoch=29-step=2400.ckpt"
    model = ViTSupCon.load_from_checkpoint(checkpoint_path).cuda()
    model.eval()
    all_features = []
    all_labels = []
    from tqdm import tqdm

    with torch.no_grad():
        for batch in tqdm(data_module.val_dataloader()):
            images, labels = batch
            images = images.cuda()
            features = model(images)
            all_features.append(features.cpu())
            all_labels.extend(labels.numpy().tolist())

    embeddings = torch.cat(all_features, dim=0).numpy()
    print(embeddings.shape)
    print(len(all_labels))
    all_labels = [classes[i].split("__")[0] for i in all_labels]

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns

    X = TSNE(n_components=2, perplexity=20.0, n_iter=1000).fit_transform(embeddings)  # compute t-SNE representation
    sns.set(font="serif", style="ticks", rc={"figure.figsize": (10, 10)})
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=all_labels,
        palette="Set2_r",
    )
    # Save figure
    plt.savefig('embedding_vis.png', dpi=300)
