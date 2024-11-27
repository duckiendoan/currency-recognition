import os

os.environ['HF_HUB_CACHE'] = 'huggingface'
import time
import itertools
import argparse
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything

from datasets import read_dataset, CustomDataModule
from models import CurrencyClassifier, ResNet50SupCon, ViTSupCon
from utils import TwoCropTransformAlbumentation


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for model and seed.")
    parser.add_argument(
        "--model",
        choices=["resnet50", "vit-small"],
        default="vit-small",
        help="Choose between 'resnet50' or 'vit-small'. Default is 'vit-small'."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Provide an integer seed value. Default is 42."
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)

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
    num_classes = len(classes)
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

    BATCH_SIZE = 128
    LEARNING_RATE = 5e-5

    data_module = CustomDataModule(train_df, valid_df, classes, TwoCropTransformAlbumentation(train_transform),
                                   val_transform, batch_size=BATCH_SIZE)
    checkpoint_path = "./currency-recognition/zh732ypf/checkpoints/epoch=29-step=2400.ckpt"

    if args.model == 'resnet50':
        encoder = ResNet50SupCon.load_from_checkpoint(checkpoint_path)
        input_dim = encoder.model.fc.in_features
        encoder.model.fc = nn.Identity()
    else:
        encoder = ViTSupCon.load_from_checkpoint(checkpoint_path)
        input_dim = encoder.model.head.in_features
        encoder.model.head = nn.Identity()

    model = CurrencyClassifier(encoder, input_dim, num_classes, learning_rate=LEARNING_RATE)
    exp_name = f"{model.__class__.__name__}__{int(time.time())}"
    wandb_logger = WandbLogger(name=exp_name, project='currency-recognition', save_code=True)
    # Train the model
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = L.Trainer(
        max_epochs=30,
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=[
            lr_monitor,
            EarlyStopping(monitor="val_f1", mode="max")
        ]
        # gradient_clip_val=2.0
    )
    trainer.fit(model, datamodule=data_module)
