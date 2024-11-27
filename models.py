import lightning as L
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet50
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score

from losses import SupConLoss

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
        scheduler = CosineAnnealingLR(optimizer, T_max=10)  # T_max is the number of epochs for a full cycle
        # total_steps = self.trainer.estimated_stepping_batches
        return [optimizer], [scheduler]


class ResNet50SupCon(L.LightningModule):
    def __init__(self, embedding_dim=128, learning_rate=5e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet50(weights='DEFAULT')
        # for param in self.model.parameters():
        #     param.requires_grad = False
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, embedding_dim)
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
        scheduler = CosineAnnealingLR(optimizer, T_max=10)  # T_max is the number of epochs for a full cycle
        # total_steps = self.trainer.estimated_stepping_batches
        return [optimizer], [scheduler]


class CurrencyClassifier(L.LightningModule):
    def __init__(self, encoder, input_dim, num_classes, learning_rate=2e-5):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.lr = learning_rate
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average='weighted')

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        val_loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.accuracy.update(preds, labels)
        self.f1.update(preds, labels)
        self.log("val_loss", val_loss)
        return val_loss

    def on_validation_epoch_end(self):
        acc = self.accuracy.compute()
        f1_score = self.f1.compute()
        self.log("val_accuracy", acc)
        self.log("val_f1", f1_score)
        self.accuracy.reset()
        self.f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]