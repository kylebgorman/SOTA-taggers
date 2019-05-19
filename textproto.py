"""Utility methods for textproto IO."""

import logging
from typing import IO

from google.protobuf import text_format

import tagdata_pb2


def read_sentences(source: IO) -> tagdata_pb2.Sentences:
    """Reads Sentences textproto from file handle."""
    sentences = tagdata_pb2.Sentences()
    text_format.ParseLines(source, sentences)
    logging.debug("Read %d sentences", len(sentences.sentences))
    return sentences


def write_sentences(sentences: tagdata_pb2.Sentences, sink: IO) -> None:
    """Writes Sentences textproto to file handle."""
    text_format.PrintMessage(sentences, sink, as_utf8=True)
    logging.debug("Wrote %d sentences", len(sentences.sentences))
