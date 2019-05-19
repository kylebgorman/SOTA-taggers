"""Accuracy helpers."""

from typing import FrozenSet, Iterator, Tuple

import confint
import tagdata_pb2

AccuracyTriple = Tuple[float, float, float]


def _tag_pairs(
    gold: tagdata_pb2.Sentences, hypo: tagdata_pb2.Sentences
) -> Iterator[Tuple[str, str]]:
    """Yield pairs of tags.

    Args:
        gold: A Sentences proto with gold tags.
        hypo: A Sentences proto with hypothesized tags.
  
    Yields:
        (gold tag, hypo tag) pairs.
    """
    assert len(gold.sentences) == len(hypo.sentences), "Mismatched lengths"
    gen = zip(gold.sentences, hypo.sentences)
    for (gold_sentence, hypo_sentence) in gen:
        assert len(gold_sentence.tags) == len(
            hypo_sentence.tags
        ), "Mismatched lengths"
        yield from zip(gold_sentence.tags, hypo_sentence.tags)


def token_accuracy(
    gold: tagdata_pb2.Sentences, hypo: tagdata_pb2.Sentences
) -> AccuracyTriple:
    """Computes token accuracy and Wilson score 95% confidence intervals.

    Args:
        gold: A Sentences proto with gold tags.
        hypo: A Sentences proto with hypothesized tags.

    Returns:
        A (token accuracy, 95% lower confidence interval, 95% upper confidence
            interval) triple.
    """
    # Computes accuracy.
    correct = 0
    total = 0
    for (gold_tag, hypo_tag) in _tag_pairs(gold, hypo):
        correct += gold_tag == hypo_tag
        total += 1
    accuracy = correct / total
    return (accuracy, *confint.wilson_score(accuracy, total))


def _token_and_tag_pairs(
    gold: tagdata_pb2.Sentences, hypo: tagdata_pb2.Sentences
) -> Iterator[Tuple[str, str, str]]:
    """Yields (token, gold tag, hypo tag) triples.

    Args:
        gold: A Sentences proto with gold tags.
        hypo: A Sentences proto with hypothesized tags.
  
    Yields:
        (token, gold tag, hypo tag) pairs.
    """
    assert len(gold.sentences) == len(hypo.sentences), "Mismatched lengths"
    gen = zip(gold.sentences, hypo.sentences)
    for (gold_sentence, hypo_sentence) in gen:
        assert len(gold_sentence.tokens) == len(
            gold_sentence.tags
        ), "Mismatched lengths"
        assert len(gold_sentence.tags) == len(
            hypo_sentence.tags
        ), "Mismatched lengths"
        yield from zip(
            gold_sentence.tokens, gold_sentence.tags, hypo_sentence.tags
        )


def oov_accuracy(
    lexicon: FrozenSet[str],
    gold: tagdata_pb2.Sentences,
    hypo: tagdata_pb2.Sentences,
) -> AccuracyTriple:
    """Computes OOV accuracy and Wilson score 95% confidence intervals.

    Args:
        train: A Sentences proto with training data.
        gold: A Sentences proto with gold tags.
        hypo: A Sentences proto with hypothesized tags.

    Returns:
        A (OOV accuracy, 95% lower confidence interval, 95% upper confidence
            interval triple).
    """
    correct = 0
    total = 0
    for (token, gold_tag, hypo_tag) in _token_and_tag_pairs(gold, hypo):
        if token in lexicon:
            continue
        correct += gold_tag == hypo_tag
        total += 1
    accuracy = correct / total
    return (accuracy, *confint.wilson_score(accuracy, total))


def sentence_accuracy(
    gold: tagdata_pb2.Sentences, hypo: tagdata_pb2.Sentences
) -> AccuracyTriple:
    """Computes sentence accuracy and Wilson score 95% confidence intervals.

    Args:
        gold: A Sentences proto with gold tags.
        hypo: A Sentences proto with hypothesized tags.

    Returns:
        A (sentence accuracy, 95% lower confidence interval, 95% upper
            confidence interval) triple.
    """
    correct = 0
    total = 0
    assert len(gold.sentences) == len(hypo.sentences), "Mismatched lengths"
    gen = zip(gold.sentences, hypo.sentences)
    for (gold_sentence, hypo_sentence) in gen:
        assert len(gold_sentence.tags) == len(
            hypo_sentence.tags
        ), "Mismatched lengths"
        gen = zip(gold_sentence.tags, hypo_sentence.tags)
        if all(gold_tag == hypo_tag for (gold_tag, hypo_tag) in gen):
            correct += 1
        total += 1
    accuracy = correct / total
    return (accuracy, *confint.wilson_score(accuracy, total))
