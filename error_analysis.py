#!/usr/bin/env python
"""Conducts a voting error analysis by comparing several tagger outputs."""

import argparse
import collections
import csv
import logging
import math
import os
import sys

from typing import Counter, Dict, List, Tuple

import numpy
from nltk.metrics.agreement import AnnotationTask
import tagdata_pb2
import textproto


class TagAnalysis:
    """Stores the analysis of a token's tagging by multiple taggers."""

    # TODO(bedricks): Add instance methods for computing tag-level agreement
    # statistics, etc.

    def __init__(self, token: str, gold_tag: str, hyps: List[str]):
        self.token = token
        self.ground_truth = gold_tag
        self.hypotheses = hyps  # Order is assumed to be relevant.

    def hypothesis_counts(self) -> Counter[str]:
        return collections.Counter(self.hypotheses)

    def tag_count(self) -> int:
        """Returns the number of tags assigned to this token."""
        return len(self.hypothesis_counts())

    def is_discordant(self) -> bool:
        """Returns whether the hypotheses disagree or not."""
        return self.tag_count() > 1

    def percent_correct(self) -> float:
        """Returns the percent of hypotheses correct for this token."""
        num_correct = 0
        for h in self.hypotheses:
            if h == self.ground_truth:
                num_correct += 1
        return num_correct / len(self.hypotheses)

    def is_correct(self) -> bool:
        """Returns whether all of the hypotheses match ground truth."""
        return math.isclose(self.percent_correct(), 1.0)

    def __repr__(self):
        return (
            f"{self.token} / {self.ground_truth}, "
            f"tags: {dict(self.hypothesis_counts())}"
        )


class SentenceAnalysis:
    """Stores the analysis of a sentence's tagging by multiple taggers."""

    def __init__(
        self, sentence: tagdata_pb2.Sentence, tagger_lookup: List[str]
    ):
        self.sentence = sentence
        self.analyzed_tokens: List[TagAnalysis] = []
        self.tagger_lookup = tagger_lookup

    def discordant_tokens(self) -> List[TagAnalysis]:
        """Returns tokens for which hypothesized tags differ."""
        return [
            self.analyzed_tokens[i] for i in self.discordant_token_indices()
        ]

    def discordant_token_indices(self) -> List[int]:
        return [
            t_i
            for t_i, tok in enumerate(self.analyzed_tokens)
            if tok.is_discordant()
        ]

    def correct_tokens(self) -> List[TagAnalysis]:
        """Returns tokens where at least one of the taggers was wrong."""
        return [t for t in self.analyzed_tokens if t.is_correct()]

    def incorrect_tokens(self) -> List[TagAnalysis]:
        return [t for t in self.analyzed_tokens if not t.is_correct()]

    def token_count(self) -> int:
        return len(self.sentence.tokens)

    def capitalization_proportion(self) -> float:
        """Computes the proportion of capitalized tokens."""
        capitalized_count = 0
        for t in self.sentence.tokens:
            if t[0].isalpha() and t[0].isupper():
                capitalized_count += 1
        if self.token_count() == 0:
            return 0.0
        return capitalized_count / self.token_count()

    def krippendorf_alpha(self) -> float:
        """Computes Krippendorff's alpha."""
        ann_data = self._annotation_task()
        return ann_data.alpha()

    def multi_kappa(self) -> float:
        """Compute what NLTK calls multi-kappa."""
        # There is a divide-by-zero bug in NLTK's implementation, so we are
        # doing this is a slightly more complex way than is normally necessary.
        task = self._annotation_task()
        Ae = task._pairwise_average(task.Ae_kappa)
        return 1.0 if Ae == 1.0 else self._annotation_task().multi_kappa()

    def _annotation_task(self) -> AnnotationTask:
        """Creates an NLTK AnnotationTask.

        Creates an NLTK AnnotationTask object for this set of hypotheses,
        initialized with an iterable of (coder, item, value) triples.
        """
        triples = []
        for (item_idx, t) in enumerate(self.analyzed_tokens):
            for (coder_idx, code) in enumerate(t.hypotheses):
                triples.append((coder_idx, item_idx, code))
        return AnnotationTask(data=triples)

    def __repr__(self):
        """Pretty-print this sentence.

        It can be pasted directly into Excel for easier manual inspection.

        Example:

        	0	1	2	3
            Heard	that	before	?
        gold:	VBN	DT	RB	.
        Collins	RB	DT	IN	.
        Flair	VBN	IN	RB	.
        NLP4J	VBN	IN	RB	.
        Stanford	VBN	IN	IN	.
        LAPOS	NNP	DT	IN	.
        TnT	VBN	IN	RB	.
        """
        to_ret = ""
        # First line: token indices.
        to_ret += "\t"
        to_ret += "\t".join([str(i) for i in range(len(self.sentence.tokens))])
        to_ret += "\n"
        # Second line: tokens themselves.
        to_ret += "\t"
        to_ret += "\t".join(self.sentence.tokens)
        to_ret += "\n"
        # Third line: gold tags.
        to_ret += "gold:\t"
        to_ret += "\t".join(self.sentence.tags)
        to_ret += "\n"
        hypothesized_tag_seq = zip(
            *[a.hypotheses for a in self.analyzed_tokens]
        )
        for (h_idx, h) in enumerate(hypothesized_tag_seq):
            to_ret += f"{self.tagger_lookup[h_idx]}\t"
            to_ret += "\t".join(h)
            to_ret += "\n"
        return to_ret


def top_k_argmax(arr: numpy.ndarray, k: int):
    # From: https://github.com/roeeaharoni/dynmt-py/blob/master/src/common.py.
    k = min(k, arr.size)
    indices = numpy.argpartition(arr, -k)[-k:]
    indices = indices[numpy.argsort(arr[indices])]
    return numpy.flip(indices, 0)


def top_k_argmin(arr: numpy.ndarray, k: int):
    # From: https://github.com/roeeaharoni/dynmt-py/blob/master/src/common.py.
    k = min(k, arr.size)
    indices = numpy.argpartition(arr, k)[:k]
    return indices[numpy.argsort(arr[indices])]


def _analyze_sentence(
    tagger_lookup: List[str],
    gold: tagdata_pb2.Sentence,
    *hypotheses: tagdata_pb2.Sentence,
) -> SentenceAnalysis:
    """Compares a set of hypothesis to a gold reference.

    This produces, for each token, a frequency count of the various
    corresponding tags in the hypotheses. So, if the gold standard was:

    the/DT red/JJ dog/NN

    And the hypotheses were (somewhat implausibly):

    the/VB red/JJ dog/NN
    the/DT red/NN dog/NN

    The result would be something like this:

    [{VB: 1, DT: 1}, {JJ: 1, NN: 1}, {NN: 2}]

    (Though note we return TagAnalysis objects, not dictionaries.)
    """
    to_ret = SentenceAnalysis(tagger_lookup=tagger_lookup, sentence=gold)
    assert all(
        len(gold.tags) == len(hyp.tags) for hyp in hypotheses
    ), "Mismatched tag counts"
    for (t_idx, (gold_tok, gold_tag)) in enumerate(
        zip(gold.tokens, gold.tags)
    ):
        hyp_tags = [hyp.tags[t_idx] for hyp in hypotheses]
        tok_analysis = TagAnalysis(
            token=gold_tok, gold_tag=gold_tag, hyps=hyp_tags
        )
        to_ret.analyzed_tokens.append(tok_analysis)
    return to_ret


def _parse_tagger_name(fname: str) -> str:
    """Extracts tagger name from a path.

    E.g., for "random_splits/1729/dev.Collins.textproto" we get "Collins".
    """
    basename = os.path.basename(fname)
    return basename.split(".")[1]


def main(args):
    # Verifies that our gold standard exists.
    if not os.path.exists(args.gold):
        logging.error("Gold file %s not found", args.gold)
        sys.exit(1)
    # Now verifies that those files exist.
    for hypo in args.hypo:
        if not os.path.exists(hypo):
            logging.error("Input file %s not found", hypo)
            sys.exit(1)
    logging.info("Gold standard: %s", args.gold)
    logging.info("Hypotheses:")
    for hypo in args.hypo:
        logging.info("\t%s", hypo)
    logging.info(f"Output directory will be %s", args.output_directory)
    with open(args.gold, "r") as source:
        gold_sentences = textproto.read_sentences(source)
    hypotheses: Dict[str, tagdata_pb2.Sentences] = {}
    for hypo in args.hypo:
        with open(hypo, "r") as source:
            this_hypo = textproto.read_sentences(source)
        hypotheses[hypo] = this_hypo
    logging.info(
        f"Ground truth has %d sentences", len(gold_sentences.sentences)
    )
    for (h_name, h_sent) in hypotheses.items():
        logging.info(
            "Hypothesis %s has %d sentences", h_name, len(h_sent.sentences)
        )
    # Sanity checks.
    h_lengths = [len(h.sentences) for h in hypotheses.values()]
    if len(set(h_lengths)) > 1:
        logging.error(
            "Hypotheses are of different sentence lengths! %r", h_lengths
        )
        sys.exit(1)
    if len(gold_sentences.sentences) != list(set(h_lengths))[0]:
        logging.error(
            "Different number of ground truth sentences than hypothesis sentences"
        )
        sys.exit(1)
    all_analyses = []
    idx = 0
    # An iterator over tuples of Sentence objects.
    hypo_sentences = zip(*[list(h.sentences) for h in hypotheses.values()])
    # Computes a lookup table of tagger names.
    tagger_lookup = [_parse_tagger_name(t) for t in args.hypo]
    for gold_sentence, h_sents in zip(
        gold_sentences.sentences, hypo_sentences
    ):
        this_analysis = _analyze_sentence(
            tagger_lookup, gold_sentence, *h_sents
        )
        all_analyses.append(this_analysis)
    discordant_tok_count = 0
    incorrect_tok_count = 0
    all_tok_count = 0
    alpha_vals = []
    kappa_vals = []
    below_thresh_indices: List[int] = []
    for idx, s in enumerate(all_analyses):
        all_tok_count += len(s.sentence.tokens)
        discordant_tok_count += len(s.discordant_tokens())
        incorrect_tok_count += len(s.incorrect_tokens())
        this_alpha = s.krippendorf_alpha()
        if this_alpha < args.interesting_score:  # Save this index somewhere.
            below_thresh_indices.append(idx)
        alpha_vals.append(this_alpha)
        kappa_vals.append(s.multi_kappa())
        print(
            f"Analyzed {all_tok_count} tokens; {discordant_tok_count} "
            f"discordant ({discordant_tok_count / all_tok_count:.4f})"
        )
    print(
        "At least one tagger incorrect: "
        f"{incorrect_tok_count / all_tok_count:.4f}"
    )
    os.makedirs(args.output_directory, exist_ok=True)
    # Prints a master file containing alpha scores and other stats for each sentence.
    with open(
        os.path.join(args.output_directory, "all_sentences_alpha_vals.csv"),
        "w",
    ) as sink:
        writer = csv.writer(sink)
        col_headers = [
            "idx",
            "length",
            "capital_proportion",
            "alpha",
            "multi_kappa",
            "tokens",
        ]
        writer.writerow(col_headers)
        for (sentence_idx, a) in enumerate(alpha_vals):
            this_sent = all_analyses[sentence_idx]
            fields = [
                sentence_idx,
                this_sent.token_count(),
                this_sent.capitalization_proportion(),
                a,
                kappa_vals[sentence_idx],
                " ".join(this_sent.sentence.tokens),
            ]
            writer.writerow(fields)
    # Grabs the lowest-scoring (lowest-agreement) sentences for some detailed analysis.
    # TODO(bedricks): Parameterize this.
    n_lowest_alpha_idx = top_k_argmin(numpy.array(alpha_vals), 100)
    sentence_detail_dirname = os.path.join(
        args.output_directory, "sentence_detail"
    )
    os.makedirs(sentence_detail_dirname, exist_ok=True)
    for (idx, x) in enumerate(n_lowest_alpha_idx):
        with open(
            os.path.join(sentence_detail_dirname, f"{idx}_{x}.txt"), "w"
        ) as this_sent_file:
            this_analysis = all_analyses[x]
            this_sent_file.write(str(this_analysis))
    # Now does interesting threshold.
    logging.info("Saving %d interesting sentences", len(below_thresh_indices))
    interesting_dir = os.path.join(
        args.output_directory, "interesting_sentences"
    )
    os.makedirs(interesting_dir, exist_ok=True)
    for idx in below_thresh_indices:
        with open(
            os.path.join(interesting_dir, f"{idx}_{alpha_vals[idx]}.txt"), "w"
        ) as sink:
            sink.write(str(all_analyses[idx]))
    # Handles any specific one-off sentence indices that were requested by the user
    if args.specific_sentence_indices:
        target_sentence_dir = os.path.join(
            args.output_directory, "target_sentences"
        )
        os.makedirs(target_sentence_dir)
        logging.info("Dumping debug info for specific sentences")
        for i in args.specific_sentence_indices:
            logging.info("Sentence %d...", i)
            this_alpha = alpha_vals[i]
            this_analysis = all_analyses[i]
            with open(
                os.path.join(target_sentence_dir, f"{i}_{this_alpha}.txt"), "w"
            ) as sink:
                sink.write(str(this_analysis))


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold", required=True, help="path to textproto of gold tag data"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="number of disagreements that are required to report",
    )
    parser.add_argument(
        "--interesting_score",
        type=float,
        default=0.9,
        help="alpha threshold below which the sentence is saved",
    )
    parser.add_argument(
        "--specific_sentence_indices",
        type=int,
        nargs="+",
        help="specific sentence indices to dump (for debugging)",
    )
    parser.add_argument(
        "--output_directory",
        required=True,
        help="name of directory containing outputf files",
    )
    parser.add_argument(
        "hypo", nargs="+", help="path to textprotos of hypothesized tag data"
    )
    main(parser.parse_args())
