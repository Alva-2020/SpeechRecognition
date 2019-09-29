"""
Process on THCHS30 and AISHELL Data downloaded.
"""

import argparse
from utils.data import THCHS30Loader, AiShellLoader, stream_write


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thchs30", type=str, help="where to load thchs source and correction file", nargs="*")
    parser.add_argument("--aishell", type=str, help="where to load aishell source and correction file", nargs="*")
    parser.add_argument("--output", type=str, help="where to save result")
    args = parser.parse_args()

    thchs30_loader = THCHS30Loader(args.thchs30)
    aishell_loader = AiShellLoader(args.aishell)

    stream = [thchs30_loader.data_it, aishell_loader.data_it]
    stream_write(args.output, stream)
