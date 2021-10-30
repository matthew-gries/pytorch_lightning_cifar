import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class CIFARClassifierModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Strictly the inference actions
        """
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _single_eval_step(self, batch, batch_idx):
        """
        Evaluate model on val/test data and return metrics
        """
        loss_function = nn.CrossEntropyLoss()
        accuracy_function = torchmetrics.Accuracy()
        precision_function = torchmetrics.Precision()
        recall_function = torchmetrics.Recall()

        x, y = batch

        y_hat = self(x)

        loss = loss_function(y_hat, y)
        acc = accuracy_function(y_hat, y)
        prec = precision_function(y_hat, y)
        recall = recall_function(y_hat, y)

        return loss, acc, prec, recall
        

    def training_step(self, batch, batch_idx):
        loss_function = nn.CrossEntropyLoss()

        x, y = batch
        y_hat = self(x)

        loss = loss_function(y_hat, y)

        self.log("Training loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    # def validation_step(self, batch, batch_idx):
    #     loss, acc, prec, recall = self._single_eval_step(batch, batch_idx)
    #     metrics = {
    #         "validation_loss": loss,
    #         "validation_accuracy": acc,
    #         "validation_precision": prec,
    #         "validation_recall": recall
    #     }
    #     self.log_dict(metrics)
    #     return metrics


    def test_step(self, batch, batch_idx):
        loss, acc, prec, recall = self._single_eval_step(batch, batch_idx)
        metrics = {
            "test_loss": loss,
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": recall
        }
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return metrics


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer