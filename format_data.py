#!/usr/bin/env python
"""Convert between data formats."""

import argparse
import logging
import nltk
import os.path
import sys

from typing import Iterator, List

import tagdata_pb2
import textproto


# Helpers.


def _horizontal_data(source) -> Iterator[str]:
    """Generator for data separated by whitespace, with possible empty lines.

    Args:
        source: File handle open for reading.

    Yields:
        The lines, with empty lines removed.
    """
    for line in source:
        line = line.strip()
        if not line:
            continue
        yield line


def _vertical_data(source) -> Iterator[List[str]]:
    """Generator for data separated by empty lines.

    Args:
        source: File handle open for reading.

    Yields:
        The empty-line separated lines, with newlines removed.
    """
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


# Importable worker functions.


def sentences_input(path: str, sourcefmt: str) -> tagdata_pb2.Sentences:
    """Reads in Sentences data from various sources.

    Args:
        path: Input file path.
        sourcefmt: Input format.

    Returns:
        Sentences data.
    """
    with open(path, "r") as source:
        if sourcefmt == "textproto":
            sentences = textproto.read_sentences(source)
        else:
            sentences = tagdata_pb2.Sentences()
            if sourcefmt == "horizontal":
                for line in _horizontal_data(source):
                    tokens = line.split()
                    sentence = sentences.sentences.add()
                    sentence.tokens.extend(tokens)
            elif sourcefmt == "vertical":
                for tokens in _vertical_data(source):
                    sentence = sentences.sentences.add()
                    sentence.tokens.extend(tokens)
            elif sourcefmt == "horizontal_tagged":
                for line in _horizontal_data(source):
                    gen = (
                        nltk.tag.util.str2tuple(token_tag)
                        for token_tag in line.split()
                    )
                    (tokens, tags) = zip(*gen)
                    sentence = sentences.sentences.add()
                    sentence.tokens.extend(tokens)
                    sentence.tags.extend(tags)
            elif sourcefmt == "vertical_tagged":
                for tokens_tags in _vertical_data(source):
                    gen = (
                        token_tag.split(None, 1) for token_tag in tokens_tags
                    )
                    (tokens, tags) = zip(*gen)
                    sentence = sentences.sentences.add()
                    sentence.tokens.extend(tokens)
                    sentence.tags.extend(tags)
            # Else unreachable.
    return sentences


def sentences_output(
    sentences: tagdata_pb2.Sentences, sinkfmt: str, path=None
) -> None:
    """Writes out in Sentences data.

    Args:
        sentences: Sentences data.
        sinkfmt: Output format.
        path: Output path.
    """
    with (sys.stdout if path is None else open(path, "w")) as sink:
        if sinkfmt == "textproto":
            textproto.write_sentences(sentences, sink)
        elif sinkfmt == "horizontal":
            for sentence in sentences.sentences:
                print(" ".join(sentence.tokens), file=sink)
        elif sinkfmt == "vertical":
            for sentence in sentences.sentences:
                for token in sentence.tokens:
                    print(token, file=sink)
                print(file=sink)
        elif sinkfmt == "horizontal_tagged":
            for sentence in sentences.sentences:
                assert sentence.tags, "No tags found"
                assert len(sentence.tokens) == len(
                    sentence.tags
                ), "Mismatched lengths"
                gen = (
                    nltk.tag.util.tuple2str(token_tag)
                    for token_tag in zip(sentence.tokens, sentence.tags)
                )
                print(" ".join(gen), file=sink)
        elif sinkfmt == "vertical_tagged":
            for sentence in sentences.sentences:
                assert sentence.tags, "No tags found"
                assert len(sentence.tokens) == len(
                    sentence.tags
                ), "Mismatched lengths"
                for (token, tag) in zip(sentence.tokens, sentence.tags):
                    print(f"{token}\t{tag}", file=sink)
                print(file=sink)
        # Else unreachable.


def convert(input_path: str, sourcefmt: str, sinkfmt: str) -> str:
    """Converts data, handling extensions.

    Args:
        input_path: Input file path.
        sourcefmt: Input format.
        sinkfmt: Output format.

    Returns:
        Output file path.
    """
    assert input_path.endswith(sourcefmt), (input_path, sourcefmt)
    output_path = input_path[: -len(sourcefmt)] + sinkfmt
    assert output_path.endswith(sinkfmt), (output_path, sinkfmt)
    logging.debug("Converting %s -> %s", input_path, output_path)
    sentences = sentences_input(input_path, sourcefmt)
    sentences_output(sentences, sinkfmt, output_path)
    return output_path


def main(args):
    convert(args.path, args.sourcefmt, args.sinkfmt)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    format_choices = (
        "textproto",
        "horizontal",
        "vertical",
        "horizontal_tagged",
        "vertical_tagged",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sourcefmt",
        default="textproto",
        choices=format_choices,
        help="input format",
    )
    parser.add_argument(
        "--sinkfmt",
        default="textproto",
        choices=format_choices,
        help="output format",
    )
    parser.add_argument("path", help="Input file path")
    main(parser.parse_args())
