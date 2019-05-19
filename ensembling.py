"""Helpers for ensembling."""

from typing import Callable, Iterable, Iterator

import tagdata_pb2
import textproto


def ensemble_sentences(
    all_sentences: Iterable[tagdata_pb2.Sentences],
    ensembling_fnc: Callable[[Iterable[str]], str],
) -> tagdata_pb2.Sentences:
    """Applies ensembling over an iterable of tagdata_pb2.Sentences messages.
  
    This takes an iterable of tagdata_pb2.Sentences messages as inputs and
    produces a new tagdata_pb2.Sentences message as an output, using the
    `ensembling_fnc` function to perform ensembling at the tag level, and then
    returns a new tagdata_pb2.Sentences message containing the ensembled
    taggings. The `tokens` field of all internal tagdata_pb2.Sentence are
    ignored.

    Args:
        all_sentences: An iterable of tagdata_pb2.Sentences messages.
        ensembling_fnc: A ensembling function to invoke at the tag level.

    Returns:
        A tagdata_pb2.Sentences message.
    """
    inner_sentences = list(sentences.sentences for sentences in all_sentences)
    assert all(
        len(inner_sentences[0]) == len(sentences)
        for sentences in inner_sentences[1:]
    ), "Mismatched lengths"
    ensembled_sentences = tagdata_pb2.Sentences()
    for all_sentence in zip(*inner_sentences):
        inner_tags = list(sentence.tags for sentence in all_sentence)
        assert all(
            len(inner_tags[0]) == len(tags) for tags in inner_tags
        ), "Mismatched lengths"
        ensembled_sentence = ensembled_sentences.sentences.add()
        for all_tags in zip(*inner_tags):
            ensembled_sentence.tags.append(ensembling_fnc(all_tags))
    return ensembled_sentences


def read_all_sentences(
    paths: Iterable[str]
) -> Iterator[tagdata_pb2.Sentences]:
    """Parses and yields Sentences textprotos.

    Args:
        paths: An iterable of paths.

    Yields:
        tagdata_pb2.Sentences messages.
    """
    for path in paths:
        with open(path, "r") as source:
            yield textproto.read_sentences(source)
