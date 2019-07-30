
import argparse
from collections import Counter
from pypinyin import Style
from pypinyin.constants import PINYIN_DICT, PHRASES_DICT
from pypinyin.core import to_fixed
from _utils.nlp.u_nlp import common_words
from deep_speech2.data_utils.utility import read_data
from typing import List, Optional


_VOCAB_COLUMN_ALIAS = {
        "pny": "pinyin",
        "han": "content"
}

_VOCAB_TYPES = list(_VOCAB_COLUMN_ALIAS.keys())


def get_vocab_column(vocab_type: str) -> str:
    if vocab_type not in _VOCAB_TYPES:
        raise ValueError("Unknown vocab type (%s), possible choices are %s" % (vocab_type, _VOCAB_TYPES))
    return _VOCAB_COLUMN_ALIAS[vocab_type]


def _fix_pny(pny: str) -> str:
    """ Modify the Mandarin pny with no tone in common popular sources. """
    pny = to_fixed(pny, strict=True, style=Style.TONE3)
    if pny[-1] not in ("1", "2", "3", "4"):  # 轻声处理
        pny += "5"
    return pny


class VocabBuilder(object):
    """
    Vocabulary builder based on given source and types.
    :param data_paths: The list of source paths. If `None`, build with common popular data source.
    :param vocab_type: The type of vocabulary to build, choices are `['pny', 'han']`.
                       If 'pny', the vocabulary is for the Mandarin pinyin,
                       If 'han', the vocabulary is for the Mandarin chars,
    :raises ValueError: If `vocab_type` is not in `['pny', 'han']`.
    """
    # TODO: Add English support.

    def __init__(self, data_paths: Optional[List[str]]=None, vocab_type: str="pny", **params):
        self.column = get_vocab_column(vocab_type.lower())
        self.data_paths = data_paths
        self.vocab_type = vocab_type
        self.params = params

    @staticmethod
    def make_pny_vocab(pny_only: bool = False) -> List[str]:
        """
        Build pny vocab based on common popular source
        :param pny_only: Whether include english letters and arabic numbers.
        :return: List of pny.
        """
        pinyin_values = ",".join(PINYIN_DICT.values()).split(",") + \
                        [pny for words_pnys in PHRASES_DICT.values() for pny_list in words_pnys for pny in pny_list]
        all_pinyin = set([_fix_pny(pny) for pny in pinyin_values])
        all_pinyin = sorted(list(all_pinyin))
        if pny_only:
            return all_pinyin
        all_numbers = [str(i) for i in range(10)]
        all_letters = [chr(x) for x in range(ord("A"), ord("Z") + 1)]
        return all_numbers + all_letters + all_pinyin  # 1635

    @staticmethod
    def make_han_vocab() -> List[str]:
        """
        Build han chars vocab based on common popular source. e.g. the `3500 common chars`
        :return: List of han chars.
        """
        return common_words()

    def _make_vocab_from_files(self, data_paths: List[str]):
        """ Make a `counter` summary on sources """
        counter = Counter()
        for path in data_paths:
            data = read_data(path, data_tag="labeled_data", to_dict=False)[self.column]
            for line in data:
                line = line.split("-") if self.vocab_type == "pny" else line
                counter.update(line)
        count_threshold = self.params.get("count_threshold", 0)
        vocab_count = [(k, count) for k, count in counter.items() if count > count_threshold]
        vocab_sorted = sorted(vocab_count, key=lambda x: x[1], reverse=True)  # sorted by count desc
        return [pair[0] for pair in vocab_sorted]

    def _make_vocab_from_common(self) -> List[str]:
        if self.vocab_type == "pny":
            return self.make_pny_vocab(False)
        elif self.vocab_type == "han":
            return self.make_han_vocab()

    def generate_vocab(self) -> List[str]:
        if self.data_paths:
            return self._make_vocab_from_files(self.data_paths)
        return self._make_vocab_from_common()


def save_vocab(vocab: List[str], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for char in vocab:
            f.write(char + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_paths", type=str, nargs="+", default=None, help="File paths for building vocabulary.")
    parser.add_argument("count_threshold", type=int, default=0, help="Truncation threshold for element counts.")
    parser.add_argument("vocab_type", type=str, choices=_VOCAB_TYPES, default="pny", help="the type of vocab.")
    parser.add_argument("vocab_path", type=str, help="File path to write the vocabulary", required=True)
    args = parser.parse_args()

    vocab_builder = VocabBuilder(
        data_paths=args.data_paths,
        vocab_type=args.vocab_type,
        count_threshold=args.count_threshold
    )

    vocab = vocab_builder.generate_vocab()
    save_vocab(vocab, args.vocab_path)
