#!/usr/bin/env python
"""Runs the Bonferroni analysis, after:

Dror, R., Baumer, G., Bogomolov, M., and Reichart, R. 2017. Replicability
analysis for natural Language processing: Testing significance with multiple
datasets. Transactions of the Association for Computational Linguistics 5:
471-486.

Namely we compute McNemar tests for all adjacent pairs across all twenty
random sets, then apply a Bonferroni correction and log how many are
significant before and after the correction."""

import argparse
import glob
import logging

from typing import Iterator, List

import accuracy
import mcnemar
import tagdata_pb2
import textproto


ASSUMED_RANKING = ["TnT", "Collins", "LAPOS", "Stanford", "NLP4J", "Flair"]


def _read_sentences(paths: Iterator[str]) -> Iterator[tagdata_pb2.Sentences]:
    for path in paths:
        with open(path, "r") as source:
            yield textproto.read_sentences(source)


def main(args):
    # Reads in gold data.
    gold_paths = sorted(glob.iglob(f"{args.data_path}/*/test.textproto"))
    gold_data = list(_read_sentences(gold_paths))
    size = len(gold_data)
    adj_alpha = args.alpha / size
    # Reads in first hypothesis data.
    hypo1_name = ASSUMED_RANKING.pop(0)
    hypo1_paths = sorted(
        glob.iglob(f"{args.data_path}/*/test.{hypo1_name}.textproto")
    )
    hypo1_data = list(_read_sentences(hypo1_paths))
    assert size == len(hypo1_data), "Mismatched lengths"
    for hypo2_name in ASSUMED_RANKING:
        # Reads in second hypothesis data.
        hypo2_paths = sorted(
            glob.iglob(f"{args.data_path}/*/test.{hypo2_name}.textproto")
        )
        hypo2_data = list(_read_sentences(hypo2_paths))
        assert size == len(hypo2_data), "Mismatched lengths"
        # Computes the number of significant McNemar tests for this data.
        significances = 0
        reorderings = 0
        for (gold, hypo1, hypo2) in zip(gold_data, hypo1_data, hypo2_data):
            # In the case that hypo1 is actually better than hypo2, we don't
            # further consider it.
            hypo1_acc = accuracy.token_accuracy(gold, hypo1)
            hypo2_acc = accuracy.token_accuracy(gold, hypo2)
            if hypo1_acc > hypo2_acc:
                reorderings += 1
                continue
            p = mcnemar.mcnemar_test(gold, hypo1, hypo2)
            if p < adj_alpha:
                significances += 1
        logging.info(
            "%s\t vs. \t%s?:\t%2d significantly better, %2d worse (%2d trials)",
            hypo1_name,
            hypo2_name,
            significances,
            reorderings,
            size,
        )
        hypo1_name = hypo2_name
        hypo1_data = hypo2_data


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="desired alpha level"
    )
    parser.add_argument("--data_path", required=True, help="path to data")
    main(parser.parse_args())
