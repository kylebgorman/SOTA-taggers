#!/usr/bin/env python
"""Flair POS tagger: prediction.

Input is assumed to be in "horizontal" format; output is in
"horizontal_tagged" format."""

# Flair 0.4.0 assumed.

import argparse
import logging

import flair
import torch

from typing import Iterator, List


def _line_reader(path) -> Iterator[List[str]]:
    with open(path, "r") as source:
        lines = []
        for line in source:
            line = line.strip()
            if line:
                lines.append(line)
            else:
                yield lines.copy()
                lines.clear()
    if lines:
        yield lines


def main(args):
    if args.require_gpu and not torch.cuda.is_available():
        logging.fatal("GPU required but not available")
        exit(1)
    tagger = flair.models.SequenceTagger.load_from_file(args.model_path)
    for tokens in _line_reader(args.predict_data_path):
        sentence = flair.data.Sentence()
        for token in tokens:
            flair_token = flair.data.Token(token)
            sentence.add_token(flair_token)
        tagger.predict(sentence)
        tokens = (token.text for token in sentence.tokens)
        tags = (token.get_tag("pos").value for token in sentence.tokens)
        for (token, tag) in zip(tokens, tags):
            print(f"{token}\t{tag}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flair POS tagger prediction")
    parser.add_argument("model_path", help="Model path")
    parser.add_argument("predict_data_path", help="Prediction data path")
    parser.add_argument(
        "--require_gpu",
        action="store_true",
        help="Require GPU training or crash",
    )
    main(parser.parse_args())
