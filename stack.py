"""Stacking ensemble for tagging."""

import pickle

import sklearn.externals.joblib
import sklearn.feature_extraction
import sklearn.linear_model

from typing import Iterable, Iterator, List

import tagdata_pb2


# Helpers.


def tags(sentences: tagdata_pb2.Sentences) -> Iterator[str]:
    for sentence in sentences.sentences:
        yield from sentence.tags


def all_tags(
    all_sentences: Iterable[tagdata_pb2.Sentence]
) -> Iterator[List[str]]:
    inner_sentences = list(sentences.sentences for sentences in all_sentences)
    assert all(
        len(inner_sentences[0]) == len(sentences)
        for sentences in inner_sentences[1:]
    ), "Mismatched lengths"
    for all_sentence in zip(*inner_sentences):
        inner_tags = list(sentence.tags for sentence in all_sentence)
        assert all(
            len(inner_tags[0]) == len(tags) for tags in inner_tags
        ), "Mismatched lengths"
        yield from zip(*inner_tags)


class Stack:
    """Stack ensembling strategy."""

    def __init__(self, *args, **kwargs):
        self.model = sklearn.linear_model.SGDClassifier(*args, **kwargs)
        self.vectorizer = sklearn.feature_extraction.DictVectorizer()

    @classmethod
    def read(cls, model_path: str):
        with open(model_path, "rb") as source:
            return sklearn.externals.joblib.load(source)

    def write(self, model_path: str):
        with open(model_path, "wb") as sink:
            sklearn.externals.joblib.dump(self, sink)

    @staticmethod
    def dictify(tags: Iterable[str]):
        """Prepares data for the vectorizer."""
        return dict(enumerate(tags))

    def fit(self, x: Iterable[Iterable[str]], y: Iterable[str]):
        vx = self.vectorizer.fit_transform([Stack.dictify(tags) for tags in x])
        self.model.fit(vx, list(y))

    # Implements the ensembling API: produces an ensemble tagging.

    def __call__(self, tags: Iterable[str]) -> str:
        vx = self.vectorizer.transform([Stack.dictify(tags)])
        return self.model.predict(vx)[0]
