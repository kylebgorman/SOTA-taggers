#!/usr/bin/env python
"""Voted ensembling for tagging."""

from typing import Iterable

import argparse
import collections
import logging
import sys

import ensembling
import textproto


def vote(all_tags: Iterable[str]) -> str:
    """Max-voting ensembling strategy.

    This implements an unweighted max-voting scheme.

    Args:
        all_tags: An iterable of hypothesized tags.

    Returns:
        The predicted tag.
    """
    counts = collections.Counter(all_tags)
    return counts.most_common(1)[0][0]


def main(args):
    all_sentences = list(ensembling.read_all_sentences(args.textproto))
    ensembled_sentences = ensembling.ensemble_sentences(all_sentences, vote)
    textproto.write_sentences(ensembled_sentences, sys.stdout)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("textproto", nargs="+", help="input .textproto files")
    main(parser.parse_args())
