def add_args(parser):
    parser.add_argument(
        "--reglogs",
        type=str,
        default="/scratch/tmp/goto80/*zst",
        # default="/scratch/hvsc/C64Music/MUSICIANS/G/Goto80/*/*/*-1.dump.zst",
        # default="/scratch/hvsc/C64Music/MUSICIANS/H/Hubbard_Rob/*/*/*-1.dump.zst",
        # default="/scratch/hvsc/C64Music/MUSICIANS/J/Jammer/Grid_Runner/1/Grid_Runner-1.dump.zst",
    )
    parser.add_argument("--model_state", type=str, default="/scratch/tmp/model.pth")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--max-sequence-length", type=int, default=2048 * 10)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--wav", type=str, default="/scratch/tmp/sidtableflip.wav")
    parser.add_argument("--csv", type=str, default="/scratch/tmp/sidtableflip.csv")
    parser.add_argument("--output-cycles", type=int, default=60e6)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--max-files", type=int, default=64)
    parser.add_argument("--diffq", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--token-csv", type=str, default=None)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--embed", type=int, default=256)
    parser.add_argument("--optimizer", type=str, default="adam")
    return parser
