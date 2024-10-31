import torch
from torch.nn import CrossEntropyLoss
from pytorch_lightning import LightningModule


class Monkey(LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = CrossEntropyLoss()
        self.args = None

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self(x)
        outputs = outputs.view(-1, self.tok_embeddings.num_embeddings)
        y_cont = y.contiguous().view(-1)
        loss = self.loss_fn(outputs, y_cont)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y_cont).float().mean()
        self.log("train_loss", loss, on_epoch=False, on_step=True)
        self.log("train_acc", acc, on_epoch=False, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        return optimizer
