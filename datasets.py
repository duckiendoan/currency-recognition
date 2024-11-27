import os
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
import lightning as L
from sklearn.model_selection import train_test_split


def read_dataset(dataset_root, dataset_name, class_rename=None, seed=42):
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
        train_df, val_df = train_test_split(dataset['train'], test_size=0.2, random_state=seed)
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