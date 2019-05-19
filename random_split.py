#!/usr/bin/env python
"""Randomly generates a 80/10/10 split for a text-format PB."""

import argparse
import logging
import random

import tagdata_pb2
import textproto


def main(args):
    logging.info("Seed: %d", args.seed)
    random.seed(args.seed)
    with open(args.input_textproto_path, "r") as source:
        sentences = textproto.read_sentences(source)
    # We have to copy into a list so as to have __setitem__.
    sentences = list(sentences.sentences)
    random.shuffle(sentences)
    length = len(sentences)
    # Train.
    shard_size = length // 10
    boundary1 = shard_size * 8
    message = tagdata_pb2.Sentences()
    message.sentences.extend(sentences[:boundary1])
    with open(args.output_train_textproto_path, "w") as sink:
        textproto.write_sentences(message, sink)
    # Dev.
    boundary2 = boundary1 + shard_size
    del message.sentences[:]
    message.sentences.extend(sentences[boundary1:boundary2])
    with open(args.output_dev_textproto_path, "w") as sink:
        textproto.write_sentences(message, sink)
    # Test.
    del message.sentences[:]
    message.sentences.extend(sentences[boundary2:])
    with open(args.output_test_textproto_path, "w") as sink:
        textproto.write_sentences(message, sink)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, help="Random seed")
    parser.add_argument(
        "--input_textproto_path", required=True, help="Input text-format PB"
    )
    parser.add_argument(
        "--output_train_textproto_path",
        required=True,
        help="Output training text-format PB (80%)",
    )
    parser.add_argument(
        "--output_dev_textproto_path",
        required=True,
        help="Output dev text-format PB (10%)",
    )
    parser.add_argument(
        "--output_test_textproto_path",
        required=True,
        help="Output test text-format PB (10%)",
    )
    main(parser.parse_args())
