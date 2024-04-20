def add_args(parser):
    parser.add_argument(
        "--reglogs",
        type=str,
        default="/scratch/hvsc/C64Music/MUSICIANS/H/Hubbard_Rob/Comman*/*/*.dump.zst",
    )
    parser.add_argument("--model_state", type=str, default="/scratch/tmp/model.pth")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--wav", type=str, default="/scratch/tmp/x.wav")
    parser.add_argument("--output_length", type=int, default=10000)
    parser.add_argument(
        "--single_model", type=str, default="/scratch/tmp/single-char.pth"
    )
    return parser
