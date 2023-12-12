import argparse
from typing import Any



def parse() -> Any:
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Model and device spec
    parser.add_argument('--device', type=str, default='cuda', help='Device to be used for training')
    parser.add_argument('--jit', action='store_true', help='Use JIT for model, weight dtypes are hardcoded')
    
    # Standard DL training params
    parser.add_argument('--mb_size', type=int, default=32, help='Mini-batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay for AdamW Optimizer')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training per experience')
    parser.add_argument('--seed', type=int, default=1234, help='Seed for randomness')
    parser.add_argument('--num_classes', type=int, default=100, help='No. of classes')
    
    parser.add_argument('--ewc_l', type=float, default=1e-2, help='Regularization param for EWC (online only)')
    parser.add_argument('--ewc_decay', type=float, default=0.4, help='EWC decay factor for online mode')
    parser.add_argument('--n_exp', type=int, default=10, help='No. of experiences in CI scenario')
    parser.add_argument('--use_ewc', action='store_true', help='Flag to use EWC regluarization in CL scenario')

    args = parser.parse_args()
    return args