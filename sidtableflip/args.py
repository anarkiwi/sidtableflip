from model import MODEL_GETTERS


def add_args(parser):
    parser.add_argument(
        "--reglogs",
        type=str,
        # default="/scratch/tmp/goto80/*zst",
        default="/scratch/hvsc/C64Music/MUSICIANS/G/Goto80/*/*/*-1.dump.zst",
        # default="/scratch/hvsc/C64Music/MUSICIANS/H/Hubbard_Rob/*/*/*-1.dump.zst",
        # default="/scratch/hvsc/C64Music/MUSICIANS/J/Jammer/Grid_Runner/1/Grid_Runner-1.dump.zst",
    )
    parser.add_argument(
        "--model_state", type=str, default="/scratch/sidtableflip/sidtableflip.pth"
    )
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--max-sequence-length", type=int, default=2048 * 10)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--wav", type=str, default="/scratch/sidtableflip/sidtableflip.wav"
    )
    parser.add_argument(
        "--csv", type=str, default="/scratch/sidtableflip/sidtableflip.csv"
    )
    parser.add_argument("--output-cycles", type=int, default=60e6)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--max-files", type=int, default=256)
    parser.add_argument("--diffq", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--token-csv", type=str, default=None)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv_heads", type=int, default=None)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--intermediate", type=int, default=None)
    parser.add_argument("--norm_eps", type=float, default=1e-5)
    parser.add_argument("--rope_base", type=int, default=10000)
    parser.add_argument("--attn_dropout", type=float, default=0)
    parser.add_argument("--model", choices=list(MODEL_GETTERS.keys()), default="llama2")
    parser.add_argument(
        "--precision", choices=["highest", "high", "medium"], default="medium"
    )
    parser.add_argument("--trainer-precision", type=str, default="bf16-mixed")
    return parser
