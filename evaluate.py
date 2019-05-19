#!/usr/bin/env python
"""Evaluates predictions."""

import argparse
import logging
import math

from typing import Iterator, Tuple

import accuracy
import mcnemar
import tagdata_pb2
import textproto


# Helpers.


def main(args):
    with open(args.gold, "r") as source:
        gold = textproto.read_sentences(source)
    with open(args.hypo1, "r") as source:
        hypo1 = textproto.read_sentences(source)
    if args.train:
        with open(args.train, "r") as source:
            train = textproto.read_sentences(source)
        # Flattens this into a single iterator of tokens.
        lexicon = frozenset(
            item for sentence in train.sentences for item in sentence.tokens
        )
    logging.info("Evaluating: %s", args.hypo1)
    logging.info(
        "Token accuracy:\t\t%.4f\t(95%% CI: %.4f, %.4f)",
        *accuracy.token_accuracy(gold, hypo1)
    )
    if args.train:
        logging.info(
            "OOV accuracy:\t\t%.4f\t(95%% CI: %.4f, %.4f)",
            *accuracy.oov_accuracy(lexicon, gold, hypo1)
        )
    logging.info(
        "Sentence accuracy:\t%.4f\t(95%% CI: %.4f, %.4f)",
        *accuracy.sentence_accuracy(gold, hypo1)
    )
    for path in args.hypo2:
        with open(path, "r") as source:
            hypo2 = textproto.read_sentences(source)
        logging.info("Evaluating: %s", path)
        logging.info(
            "Token accuracy:\t\t%.4f\t(95%% CI: %.4f, %.4f)",
            *accuracy.token_accuracy(gold, hypo2)
        )
        if args.train:
            logging.info(
                "OOV accuracy:\t\t%.4f\t(95%% CI: %.4f, %.4f)",
                *accuracy.oov_accuracy(lexicon, gold, hypo2)
            )
        logging.info(
            "Sentence accuracy:\t%.4f\t(95%% CI: %.4f, %.4f)",
            *accuracy.sentence_accuracy(gold, hypo2)
        )
        logging.info(
            "McNemar mid-p:\t\t%.4f", mcnemar.mcnemar_test(gold, hypo1, hypo2)
        )
        hypo1 = hypo2


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold", required=True, help="Path to textproto of gold tag data"
    )
    parser.add_argument(
        "--train",
        help="Path to textproto of training data, used to compute OOV accuracy",
    )
    parser.add_argument(
        "hypo1", help="Path to textproto of hypothesized tag data"
    )
    parser.add_argument(
        "hypo2",
        nargs="*",
        help="Optional additional textprotos of hypothesized tag data",
    )
    main(parser.parse_args())
