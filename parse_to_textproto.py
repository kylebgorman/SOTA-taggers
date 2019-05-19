#!/usr/bin/env python
"""Reads parse files and converts the data to text-format PBs.

This script reads POS tagging data from parsed files, outputting the data
as text-format protocol buffers. This data can then be converted into whatever
format is needed downstream.

The files may contain multiple sentences, so we count brackets to figure
out when sentences are complete.

The NLTK tree library requires that all non-terminal nodes have a string label,
but the PTB-3 files don't have a label for the topmost node in a sentence.
Therefore we insert one, "TOP" by default."""

import argparse
import nltk
import sys

from typing import Iterable, Iterator, List, Tuple

import tagdata_pb2
import textproto


Pair = Tuple[str, str]


BRACKETS = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
}


# Set this to False to disable bracket conversion.
ENABLE_BRACKET_FIXES = False


def _tag_fixes(pairs: Iterable[Pair]) -> Iterator[Pair]:
    """Fixes tags and removes empty nodes."""
    for (token, tag) in pairs:
        if tag == "-NONE-":
            continue
        if ENABLE_BRACKET_FIXES:
            token = BRACKETS.get(token, token)
            tag = BRACKETS.get(tag, tag)
        yield (token, tag)


def _parse_to_pos(
    parse_path: str, top_node: str = "TOP"
) -> Iterator[List[Pair]]:
    """Generator for extracting (token, POS tag) tuples from .parse files.

    Args:
        parse_path: File path to the .parse file to process.
        top_node: Label to use for the inserted top node (default: "TOP").

    Yields:
        Lists of (token, POS tag) tuples.
    """
    # This is hackish in various ways but it definitely works.
    brackets = 0
    buf = []
    with open(parse_path, "r") as source:
        for line in source:
            line = line.strip()
            for char in line:
                if char == "(":
                    brackets += 1
                elif char == ")":
                    brackets -= 1
            if line:
                buf.append(line)
            if buf and not brackets:
                assert buf[0].startswith("("), buf[0]
                # Handles missing "TOP" in the PTB-3 data.
                if not buf[0].startswith("(TOP"):
                    buf[0] = buf[0][0] + "TOP " + buf[0][1:]
                treestring = " ".join(buf)
                tree = nltk.tree.Tree.fromstring(treestring)
                yield list(_tag_fixes(tree.pos()))
                buf.clear()
        assert not buf, buf
        assert not brackets, brackets


def main(args):
    sentences = tagdata_pb2.Sentences()
    for parse in args.parse:
        for tokens_tags in _parse_to_pos(parse):
            sentence = sentences.sentences.add()
            (tokens, tags) = zip(*tokens_tags)
            sentence.tokens.extend(tokens)
            sentence.tags.extend(tags)
    textproto.write_sentences(sentences, sys.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parse", nargs="+", help="Input files")
    main(parser.parse_args())
