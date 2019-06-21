#!/usr/bin/env python
"""Stack ensembling for tagging."""

import argparse
import logging
import sys

import ensembling
import stack
import textproto


def main(args):
    model = stack.Stack.read(args.model_path)
    all_sentences = list(ensembling.read_all_sentences(args.textproto))
    ensembled_sentences = ensembling.ensemble_sentences(all_sentences, model)
    textproto.write_sentences(ensembled_sentences, sys.stdout)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", help="input model file")
    parser.add_argument("textproto", nargs="+", help="input .textproto files")
    main(parser.parse_args())
