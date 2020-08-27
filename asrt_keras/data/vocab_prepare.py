"""
Generate vocab
"""
import os
import argparse
from evan_utils.nlp import ALL_PNYS, LETTERS, NUMBERS


def generate_vocab_file(file: str):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    try:
        os.remove(file)
    except FileNotFoundError:
        pass
    vocab = [" "] + ALL_PNYS + LETTERS + NUMBERS + ["_"]
    assert len(vocab) == len(ALL_PNYS) + len(LETTERS) + len(NUMBERS) + 2, "Invalid vocab, there are duplicates."
    with open(file, "w", encoding="utf-8") as f:
        for s in vocab:
            f.write(s + "\n")
    print(f"Write vocab into file `{file}` successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None, help="where to save result")
    args = parser.parse_args()

    generate_vocab_file(args.output)
