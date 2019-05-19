#!/usr/bin/env python
"""Deterministically samples a trailing subset of a text-format PB."""

import argparse

import textproto


def main(args):
    with open(args.input_textproto_path, "r") as source:
        sentences = textproto.read_sentences(source)
    # Because it's a repeated message, we have to do it this way.
    sample = list(sentences.sentences[-args.size :])
    del sentences.sentences[:]
    sentences.sentences.extend(sample)
    with open(args.output_textproto_path, "w") as sink:
        textproto.write_sentences(sentences, sink)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_textproto_path", help="Input text-formant PB")
    parser.add_argument("output_textproto_path", help="Output text-format PB")
    parser.add_argument("--size", required=True, type=int, help="Sample size")
    main(parser.parse_args())
