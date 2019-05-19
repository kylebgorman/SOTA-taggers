"""McNemar test helper."""

import scipy.stats

import tagdata_pb2

from typing import Iterator, Tuple


def _tag_triples(
    gold: tagdata_pb2.Sentences,
    hypo1: tagdata_pb2.Sentences,
    hypo2: tagdata_pb2.Sentences,
) -> Iterator[Tuple[str, str, str]]:
    """Yields triples of tags.

    Args:
        gold: a Sentences proto with gold tags.
        hypo1: a Sentences proto with hypothesized tags.
        hypo2: a Sentences proto with hypothesized tags.

    Yields:
        (gold tag, hypo tag, additional hypo tag) triples.
    """
    assert (
        len(gold.sentences) == len(hypo1.sentences) == len(hypo2.sentences)
    ), "Mismatched lengths"
    gen = zip(gold.sentences, hypo1.sentences, hypo2.sentences)
    for (gold_sentence, hypo_sentence, hypo2_sentence) in gen:
        assert (
            len(gold_sentence.tags)
            == len(hypo_sentence.tags)
            == len(hypo2_sentence.tags)
        ), "Mismatched lengths"
        yield from zip(
            gold_sentence.tags, hypo_sentence.tags, hypo2_sentence.tags
        )


def mcnemar_test(
    gold: tagdata_pb2.Sentences,
    hypo1: tagdata_pb2.Sentences,
    hypo2: tagdata_pb2.Sentences,
) -> float:
    """McNemar's test (mid-p) variant. The formula is adapted from:
    
    M.W. Fagerland, S. Lydersen, P. Laake. 2013. The McNemar test for 
    binary matched-pairs data: Mid-p and asymptotic are better than exact 
    conditional. BMC Medical Research Methodology 13: 91.
  
    Args:
        gold: a Sentences proto with gold tags.
        hypo1: a Sentences proto with hypothesized tags.
        hypo2: a Sentences proto with hypothesized tags.

    Returns:
        A McNemar's mid-p-value.
    """
    wins1 = 0
    wins2 = 0
    for (gold_tag, hypo1_tag, hypo2_tag) in _tag_triples(gold, hypo1, hypo2):
        if gold_tag == hypo1_tag and gold_tag != hypo2_tag:
            wins1 += 1
        elif gold_tag != hypo1_tag and gold_tag == hypo2_tag:
            wins2 += 1
        # Else nothing.
    n = wins1 + wins2
    x = min(wins1, wins2)
    dist = scipy.stats.binom(n, 0.5)
    return 2.0 * dist.cdf(x) - dist.pmf(x)
