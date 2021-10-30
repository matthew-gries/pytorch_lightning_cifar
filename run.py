from data import get_data_loaders
from classifier import CIFARClassifierModule
import pytorch_lightning as pl


if __name__ == "__main__":

    train_loader, test_loader = get_data_loaders()
    model = CIFARClassifierModule()
    trainer = pl.Trainer()

    trainer.fit(model, train_loader)

    trainer.test(model, test_loader)