import torch
from torch.nn import CrossEntropyLoss
from pytorch_lightning import LightningModule


class Monkey(LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = CrossEntropyLoss()
        self.args = None

    @torch.compiler.disable
    def log_nocompile(self, loss, acc):
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, on_step=True, prog_bar=True)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_cont = y.contiguous().view(-1)
        outputs = self(x).view(-1, self.tok_embeddings.num_embeddings)
        loss = self.loss_fn(outputs, y_cont)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y_cont).float().mean()
        self.log_nocompile(loss, acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.args.learning_rate, fused=True
        )
        return optimizer
