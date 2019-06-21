#!/usr/bin/env python
"""Train stacking ensemble for tagging."""

import argparse
import logging
import sklearn.feature_extraction
import sklearn.linear_model

import ensembling
import stack
import textproto


def main(args):
    all_sentences = list(ensembling.read_all_sentences(args.hypo))
    with open(args.gold, "r") as source:
        gold_sentences = textproto.read_sentences(source)
    # L2-regularized log-linear classifier.
    model = stack.Stack(
        loss="log",
        penalty="l2",
        max_iter=100,
        tol=1e-3,
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(stack.all_tags(all_sentences), stack.tags(gold_sentences))
    model.write(args.model_path)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", help="output model file")
    parser.add_argument("--seed", required=True, type=int, help="random seed")
    parser.add_argument("gold", help="path to textproto of gold tag data")
    parser.add_argument(
        "hypo", nargs="+", help="path to textprotos of hypothesized tag data"
    )
    main(parser.parse_args())
