#!/usr/bin/env python
"""Computes basic statistics for a textproto."""

import argparse
import logging

import textproto


def main(args):
    with open(args.textproto, "r") as source:
        sentences = textproto.read_sentences(source)
    n_tokens = sum(len(sentence.tokens) for sentence in sentences.sentences)
    n_sentences = len(sentences.sentences)
    logging.info("%d sentences and %d tokens", n_sentences, n_tokens)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("textproto", help="Path to textproto")
    main(parser.parse_args())
