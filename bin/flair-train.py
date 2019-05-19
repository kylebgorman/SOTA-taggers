#!/usr/bin/env python
"""Flair POS tagger: training.
    
Input is assumed to be in "horizontal_tagged" format."""

# Flair 0.4.0 assumed.

import argparse
import logging

import flair
import torch

from typing import Iterator, List


## Hyperparameters from the Flair documentation.

# Model.
TAG_TYPE = "pos"
HIDDEN_SIZE = 256
USE_CRF = True

# Optimization.
LEARNING_RATE = 0.1
MINI_BATCH_SIZE = 32
MAX_EPOCHS = 150


def _line_reader(path) -> Iterator[List[str]]:
    with open(path, "r") as source:
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


def _sentence_reader(path) -> Iterator[flair.data.Sentence]:
    for lines in _line_reader(path):
        sentence = flair.data.Sentence()
        for line in lines:
            (token, tag) = line.split()
            flair_token = flair.data.Token(token)
            flair_token.add_tag(TAG_TYPE, tag)
            sentence.add_token(flair_token)
        yield sentence


def main(args):
    if args.require_gpu and not torch.cuda.is_available():
        logging.fatal("GPU required but not available")
        exit(1)
    logging.info("Reading training data")
    # These have to be lists as they'll be shuffled.
    train = list(_sentence_reader(args.train_data_path))
    dev = list(_sentence_reader(args.dev_data_path))
    test = list(_sentence_reader(args.test_data_path))
    corpus = flair.data.TaggedCorpus(train, dev, test)
    logging.info("Creating tag dictionary")
    tag_dictionary = corpus.make_tag_dictionary(TAG_TYPE)
    logging.info("Creating embeddings")
    embeddings = flair.embeddings.StackedEmbeddings(
        [
            flair.embeddings.WordEmbeddings("extvec"),
            flair.embeddings.FlairEmbeddings("news-forward"),
            flair.embeddings.FlairEmbeddings("news-backward"),
        ]
    )
    logging.info("Creating tagger")
    tagger = flair.models.SequenceTagger(
        hidden_size=HIDDEN_SIZE,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=TAG_TYPE,
        use_crf=USE_CRF,
    )
    logging.info("Creating trainer")
    trainer = flair.trainers.ModelTrainer(tagger, corpus)
    logging.info("Training tagger")
    trainer.train(
        args.model_path,
        evaluation_metric=flair.training_utils.EvaluationMetric.MICRO_ACCURACY,
        learning_rate=LEARNING_RATE,
        mini_batch_size=MINI_BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        embeddings_in_memory=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flair POS tagger training")
    parser.add_argument("train_data_path", help="Training data path")
    # This expects dev and test too, but they're not used for anything except
    # building the tag dictionaries.
    parser.add_argument("dev_data_path", help="Development data path")
    parser.add_argument("test_data_path", help="Test data path")
    parser.add_argument("model_path", help="Model path")
    parser.add_argument(
        "--require_gpu",
        action="store_true",
        help="Require GPU training or crash",
    )
    main(parser.parse_args())
