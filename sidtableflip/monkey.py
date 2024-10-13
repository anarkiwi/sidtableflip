import torch
from torch.nn import CrossEntropyLoss
from pytorch_lightning import LightningModule


class Monkey(LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = None
        self.args = None

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self(x)
        outputs = outputs.view(-1, self.tok_embeddings.num_embeddings)
        if self.loss_fn is None:
            self.loss_fn = CrossEntropyLoss()
        loss = self.loss_fn(outputs, y.contiguous().view(-1))
        # big slowdown to self.log!
        # self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
