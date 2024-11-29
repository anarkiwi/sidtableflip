import bitsandbytes as bnb
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from schedulefree import AdamWScheduleFree
from torchtune.models.gemma._component_builders import gemma
from torchtune.models.llama2._component_builders import llama2
from torchtune.models.mistral._component_builders import mistral
from torchtune.models.phi3._component_builders import phi3
from torchtune.models.qwen2._component_builders import qwen2
import torchmetrics


class SchedulerFreeModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath):
        opts = trainer.optimizers
        if not isinstance(opts, list):
            opts = [opts]

        opt_modes = [
            (opt, opt.param_groups[0]["train_mode"])
            for opt in opts
            if isinstance(opt, AdamWScheduleFree)
        ]

        for opt, train_mode in opt_modes:
            if train_mode:
                opt.eval()

        super()._save_checkpoint(trainer, filepath)

        for opt, train_mode in opt_modes:
            if train_mode:
                opt.train()


def get_gemma(n_vocab, args):
    intermediate = args.intermediate if args.intermediate else args.embed
    head_dim = args.embed // args.heads
    kv_heads = args.kv_heads if args.kv_heads else args.heads

    return gemma(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=kv_heads,
        embed_dim=args.embed,
        head_dim=head_dim,
        intermediate_dim=intermediate,
        max_seq_len=args.max_sequence_length,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


def get_llama2(n_vocab, args):
    return llama2(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        max_seq_len=args.max_sequence_length,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


def get_mistral(n_vocab, args):
    intermediate = args.intermediate if args.intermediate else args.embed

    return mistral(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        intermediate_dim=intermediate,
        max_seq_len=args.max_sequence_length,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


def get_phi3(n_vocab, args):
    intermediate = args.intermediate if args.intermediate else args.embed

    return phi3(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        intermediate_dim=intermediate,
        max_seq_len=args.max_sequence_length,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


def get_qwen2(n_vocab, args):
    intermediate = args.intermediate if args.intermediate else args.embed

    return qwen2(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        intermediate_dim=intermediate,
        max_seq_len=args.max_sequence_length,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


MODEL_GETTERS = {
    "gemma": get_gemma,
    "llama2": get_llama2,
    "mistral": get_mistral,
    "phi3": get_phi3,
    "qwen2": get_qwen2,
}


class Model(LightningModule):
    def __init__(self, args, n_vocab):
        super().__init__()
        self.args = args
        self.n_vocab = n_vocab
        self.save_hyperparameters("args", "n_vocab")
        self.model = MODEL_GETTERS[args.model](n_vocab, args)
        self.optimizer = AdamWScheduleFree(
            self.parameters(),
            lr=self.args.learning_rate,
            foreach=True,
        )
        # self.optimizer = torch.optim.AdamW(
        #    self.parameters(),
        #    lr=self.args.learning_rate,
        #    fused=True,
        #    # weight_decay=1e-1,
        # )
        # self.optimizer = bnb.optim.AdamW(
        #    self.parameters(),
        #    lr=self.args.learning_rate,
        # )
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #    self.optimizer, gamma=0.5
        # )

    @torch.compiler.disable
    def log_nocompile(self, loss, acc):
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        self.log("train_acc", acc, on_epoch=True, on_step=True)

    def training_step(self, train_batch):
        x, y = train_batch
        y_cont = y.view(-1)
        logits = self.model(x)
        outputs = logits.view(-1, logits.size(-1))
        acc = torchmetrics.functional.classification.multiclass_accuracy(
            outputs,
            y_cont,
            self.n_vocab,
            validate_args=False,
        )
        loss = torch.nn.functional.cross_entropy(outputs, y_cont)
        self.log_nocompile(loss, acc)
        return loss

    def configure_optimizers(self):
        # return [self.optimizer], [self.scheduler]
        return self.optimizer

    def set_optimizer_state(self, state):
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]

        for opt in opts:
            if isinstance(opt, AdamWScheduleFree):
                if state == "train":
                    opt.train()
                elif state == "eval":
                    opt.eval()
                else:
                    raise ValueError(f"Unknown train state {state}")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.set_optimizer_state("train")

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.set_optimizer_state("eval")


def get_model(dataset, args, mode=None, args_override=None):
    if args_override:
        for k, v in args_override.items():
            setattr(args, k, v)
    torch.set_float32_matmul_precision(args.precision)
    model = Model(args, dataset.n_vocab)
    return torch.compile(model, mode=mode)


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
