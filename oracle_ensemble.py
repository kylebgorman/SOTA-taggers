#!/usr/bin/env python
"""Oracle ensemble for tagging."""

from typing import Iterable

import argparse
import collections
import logging
import sys

import ensembling
import tagdata_pb2
import textproto


def oracle_sentences(
    gold_sentences: tagdata_pb2.Sentences,
    all_sentences: Iterable[tagdata_pb2.Sentences],
) -> tagdata_pb2.Sentences:
    """Computes an oracle given an iterable of tagdata_pb2.Sentences messages.

    This takes a tagdata_pb2.Sentences message representing a gold tagging,
    and an iterable of tagdata_pb2.Sentences messages representing hypothesized
    taggings, and produces a new tagdata_p2.Sentences message as an output.
    If at least one of the hypothesized taggings has the correct tag according to
    the gold tagging, this is used; if not, we select the most common one instead.
    This provides an upper bound for any ensembling technique.

    Args:
        gold_sentences: A tagdata_pb2.Sentences message; the gold tagging.
        all_sentences: An iterable of tagdata_pb2.Sentences messages; the
            hypothesized taggings.

    Returns:
        A tagdata_pb2.Sentences message.
    """
    inner_sentences = list(sentences.sentences for sentences in all_sentences)
    assert all(
        len(gold_sentences.sentences) == len(sentences)
        for sentences in inner_sentences
    ), "Mismatched lengths"
    ensembled_sentences = tagdata_pb2.Sentences()
    gen = zip(gold_sentences.sentences, *inner_sentences)
    correct = 0
    total = 0
    for (gold_sentence, *all_sentence) in gen:
        inner_tags = list(sentence.tags for sentence in all_sentence)
        assert all(
            len(gold_sentence.tags) == len(tags) for tags in inner_tags
        ), "Mismatched lengths"
        ensembled_sentence = ensembled_sentences.sentences.add()
        for (gold_tag, *all_tags) in zip(gold_sentence.tags, *inner_tags):
            if any(gold_tag == tag for tag in all_tags):
                ensembled_sentence.tags.append(gold_tag)
                correct += 1
            else:
                counts = collections.Counter(all_tags)
                ensembled_sentence.tags.append(counts.most_common(1)[0][0])
            total += 1
    logging.info("Accuracy (oracle):\t%.4f", correct / total)
    return ensembled_sentences


def main(args):
    with open(args.gold, "r") as source:
        gold_sentences = textproto.read_sentences(source)
    all_sentences = list(ensembling.read_all_sentences(args.hypo))
    ensembled_sentences = oracle_sentences(gold_sentences, all_sentences)
    textproto.write_sentences(ensembled_sentences, sys.stdout)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", help="path to textproto of gold tag data")
    parser.add_argument(
        "hypo", nargs="+", help="path to textproto of hypothesized tag data"
    )
    main(parser.parse_args())
