import torch
from torch.nn import CrossEntropyLoss
from pytorch_lightning import LightningModule


class Monkey(LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = CrossEntropyLoss()
        self.args = None
        self.dataset = None

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self(x)
        outputs = outputs.view(-1, self.dataset.n_vocab)
        loss = self.loss_fn(outputs, y.contiguous().view(-1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.args.learning_rate)
        return optimizer
