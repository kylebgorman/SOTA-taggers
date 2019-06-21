#!/usr/bin/env python
"""Training-prediction interfaces for the taggers."""

import abc
import argparse
import logging
import os.path
import shutil
import subprocess
import sys
import tempfile
import time

import format_data


# Should we mute STDERR?
MUTE = True
STDERR = open(os.devnull, "w") if MUTE else sys.stderr


# Helper classes.


class LogTime:
    """Context manager which logs elapsed wall clock time."""

    def __init__(self, label, logfnc=logging.info):
        self.label = label
        self.logfnc = logfnc

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end = time.time()
        self.logfnc("%s:\t%ds", self.label, self.end - self.start)


# Abstract base classes for taggers.


class TaggerInterface(abc.ABC):
    """"Abstract base class for taggers."""

    # Static method for converting data to desired format. The string
    # returned is the path to the new file, or the input file in the case
    # this was no-op.

    efmt = "textproto"

    # Concrete method for getting a sensible temp name.

    @staticmethod
    def temp_path(input_path: str, class_name: str, ifmt: str) -> str:
        (dire, name) = os.path.split(input_path)
        input_base = os.path.splitext(name)[0]
        temp_base = f"{input_base}.{class_name}.{ifmt}"
        return os.path.join(dire, temp_base)

    # Trains a model, storing temporary data in a temporary file or
    # directory.

    @abc.abstractmethod
    def __init__(self, train_path: str, dev_path: str, test_path: str) -> None:
        pass

    # Can be called on train, dev, or test.

    def predict(self, input_path: str) -> None:
        pass

    # NB: most concrete implementations will also overload __del__ to
    # delete the temporary file or directory.


class VerticalTaggerInterface(TaggerInterface):
    """"Implementation of vertical-tagged data converters."""

    ifmt = "vertical_tagged"

    @staticmethod
    def convert_to_train(data_path: str) -> str:
        return format_data.convert(data_path, "textproto", "vertical_tagged")

    @staticmethod
    def convert_to_predict(data_path: str) -> str:
        return format_data.convert(data_path, "textproto", "vertical")

    @staticmethod
    def convert_from(data_path: str) -> str:
        output_path = format_data.convert(
            data_path, "vertical_tagged", "textproto"
        )
        os.remove(data_path)
        return output_path


class HorizontalTaggerInterface(TaggerInterface):
    """"Implementation of horizontal-tagged data converters."""

    ifmt = "horizontal_tagged"

    @staticmethod
    def convert_to_train(data_path: str) -> str:
        return format_data.convert(data_path, "textproto", "horizontal_tagged")

    @staticmethod
    def convert_to_predict(data_path: str) -> str:
        return format_data.convert(data_path, "textproto", "horizontal")

    @staticmethod
    def convert_from(data_path: str) -> str:
        output_path = format_data.convert(
            data_path, "horizontal_tagged", "textproto"
        )
        os.remove(data_path)
        return output_path


# Concrete implementations of taggers.


class TnT(VerticalTaggerInterface):
    """Interface for the TnT tagger (Brants 2000)."""

    train_binary = "bin/./tnt-para"

    def __init__(self, train_path: str, dev_path: str, test_path: str) -> None:
        self.model_dir = tempfile.mkdtemp(prefix=self.__class__.__name__ + "_")
        # TnT will add extensions to this.
        self.model_path = os.path.join(self.model_dir, "model")
        train_path = TnT.convert_to_train(train_path)
        with LogTime(f"Train {train_path}"):
            subprocess.check_call(
                ("nice", self.train_binary, "-o", self.model_path, train_path),
                stderr=STDERR,
            )

    predict_binary = "bin/./tnt"

    def predict(self, input_path: str) -> None:
        input_path = TnT.convert_to_predict(input_path)
        temp_path = TnT.temp_path(
            input_path, self.__class__.__name__, self.ifmt
        )
        with open(temp_path, "w") as sink:
            with LogTime(f"Predict {input_path}"):
                subprocess.check_call(
                    (
                        "nice",
                        self.predict_binary,
                        "-v0",
                        self.model_path,
                        input_path,
                    ),
                    stdout=sink,
                )
        TnT.convert_from(temp_path)

    def __del__(self):
        shutil.rmtree(self.model_dir)


class Collins(VerticalTaggerInterface):
    """Interface for Collins' (2002) tagger (Yarmohammadi 2014)."""

    invocation = ("nice", "java", "-mx2g", "-jar", "bin/collins.jar")

    def __init__(self, train_path: str, dev_path: str, test_path: str) -> None:
        self.model_dir = tempfile.mkdtemp(prefix=self.__class__.__name__ + "_")
        train_path = Collins.convert_to_train(train_path)
        dev_path = Collins.convert_to_train(dev_path)
        with LogTime(f"Train {train_path} ({dev_path})"):
            subprocess.check_call(
                (
                    *self.invocation,
                    "-mode",
                    "train",
                    "-train",
                    train_path,
                    "-dev",
                    dev_path,
                    "-mdlpath",
                    self.model_dir,
                ),
                stderr=STDERR,
            )

    def predict(self, input_path: str) -> None:
        input_path = Collins.convert_to_predict(input_path)
        temp_path = Collins.temp_path(
            input_path, self.__class__.__name__, self.ifmt
        )
        with LogTime(f"Predict {input_path}"):
            subprocess.check_call(
                (
                    *self.invocation,
                    "-mode",
                    "test",
                    "-in",
                    input_path,
                    "-mdlpath",
                    self.model_dir,
                    "-outname",
                    temp_path,
                ),
                stderr=STDERR,
            )
        Collins.convert_from(temp_path)

    def __del__(self):
        shutil.rmtree(self.model_dir)


class LAPOS(HorizontalTaggerInterface):
    """Interface for the LAPOS tagger (Tsuruoka et al. 2011)."""

    train_binary = "bin/./lapos-learn"

    def __init__(self, train_path: str, dev_path: str, test_path: str) -> None:
        # Here the path provided is actually a directory.
        self.model_dir = tempfile.mkdtemp(prefix="LAPOS_")
        train_path = LAPOS.convert_to_train(train_path)
        with LogTime(f"Train {train_path}"):
            subprocess.check_call(
                (
                    "nice",
                    self.train_binary,
                    f"--model={self.model_dir}",
                    train_path,
                ),
                stderr=STDERR,
            )

    predict_binary = "bin/./lapos"

    def predict(self, input_path: str) -> None:
        input_path = LAPOS.convert_to_predict(input_path)
        temp_path = LAPOS.temp_path(
            input_path, self.__class__.__name__, self.ifmt
        )
        with open(temp_path, "w") as sink:
            with LogTime(f"Predict {input_path}"):
                subprocess.check_call(
                    (
                        "nice",
                        self.predict_binary,
                        f"--model={self.model_dir}",
                        input_path,
                    ),
                    stdout=sink,
                    stderr=STDERR,
                )
        LAPOS.convert_from(temp_path)

    def __del__(self):
        shutil.rmtree(self.model_dir)


class Stanford(HorizontalTaggerInterface):
    """Interface for the Stanford tagger (Toutanova et al. 2003)."""

    config = "bin/stanford.properties"
    invocation = (
        "nice",
        "java",
        "-mx3g",
        "-classpath",
        "bin/stanford-postagger.jar",
        "edu.stanford.nlp.tagger.maxent.MaxentTagger",
    )

    def __init__(self, train_path: str, dev_path: str, test_path: str) -> None:
        self.model_path = tempfile.mkstemp(
            prefix="stanford_", suffix=".model"
        )[1]
        train_path = Stanford.convert_to_train(train_path)
        with LogTime(f"Train {train_path}"):
            subprocess.check_call(
                (
                    *self.invocation,
                    "-prop",
                    self.config,
                    "-model",
                    self.model_path,
                    "-trainFile",
                    train_path,
                ),
                stderr=STDERR,
            )

    def predict(self, input_path: str) -> None:
        input_path = Stanford.convert_to_predict(input_path)
        temp_path = Stanford.temp_path(
            input_path, self.__class__.__name__, self.ifmt
        )
        with open(temp_path, "w") as sink:
            with LogTime(f"Predict {input_path}"):
                subprocess.check_call(
                    (
                        *self.invocation,
                        "-model",
                        self.model_path,
                        "-textFile",
                        input_path,
                    ),
                    stdout=sink,
                    stderr=STDERR,
                )
        Stanford.convert_from(temp_path)

    def __del__(self):
        os.remove(self.model_path)


class NLP4J(VerticalTaggerInterface):
    """Interface for NLP4J (Choi 2016)."""

    train_config = "bin/nlp4j-train.xml"
    train_script = "bin/nlp4j/appassembler/bin/./nlptrain"
    conf_template = """<configuration>
    <tsv>
        <column index="0" field="form" />
        <column index="1" field="pos" />
    </tsv>

    <lexica>
        <ambiguity_classes field="word_form_simplified_lowercase">edu/emory/mathcs/nlp/lexica/en-ambiguity-classes-simplified-lowercase.xz</ambiguity_classes>
        <word_clusters field="word_form_simplified_lowercase">edu/emory/mathcs/nlp/lexica/en-brown-clusters-simplified-lowercase.xz</word_clusters>
    </lexica>

    <models>
        <pos>{}</pos>
    </models>
</configuration>"""

    def __init__(self, train_path: str, dev_path: str, test_path: str) -> None:
        self.model_path = tempfile.mkstemp(
            prefix=self.__class__.__name__ + "_", suffix=".xz"
        )[1]
        train_path = NLP4J.convert_to_train(train_path)
        dev_path = NLP4J.convert_to_train(dev_path)
        with LogTime(f"Train {train_path} ({dev_path})"):
            subprocess.check_call(
                (
                    "nice",
                    self.train_script,
                    "-mode",
                    "pos",
                    "-c",
                    self.train_config,
                    "-t",
                    train_path,
                    "-d",
                    dev_path,
                    "-m",
                    self.model_path,
                ),
                stdout=STDERR,
                stderr=STDERR,
            )
        # Generates configuration file, which unfortunately needs to
        # contain the actual model path
        self.conf_path = tempfile.mkstemp(
            prefix=self.__class__.__name__ + "_", suffix=".xml", text=False
        )[1]
        with open(self.conf_path, "w") as sink:
            print(self.conf_template.format(self.model_path), file=sink)

    predict_script = "bin/nlp4j/appassembler/bin/./nlpdecode"

    def predict(self, input_path: str) -> None:
        # This one is still okay with the two-column format.
        input_path = NLP4J.convert_to_train(input_path)
        # This doesn't permit me to specify an output extension; instead,
        # it adds ".nlp" to whatever the input data is.
        with LogTime(f"Predict {input_path}"):
            subprocess.check_call(
                (
                    "nice",
                    self.predict_script,
                    "-c",
                    self.conf_path,
                    "-i",
                    input_path,
                    "-format",
                    "tsv",
                ),
                stdout=STDERR,
                stderr=STDERR,
            )
        internal_path = input_path + ".nlp"
        # It is now in a pseudo-CoNLL format; we convert it back to
        # vertical-tagged format.
        temp_path = NLP4J.temp_path(
            input_path, self.__class__.__name__, self.ifmt
        )
        with open(internal_path, "r") as source:
            with open(temp_path, "w") as sink:
                for line in source:
                    line = line.rstrip()
                    if not line:
                        print(file=sink)
                        continue
                    (_, token, _, tag, _) = line.split(None, 4)
                    print(f"{token}\t{tag}", file=sink)
        os.remove(internal_path)
        NLP4J.convert_from(temp_path)

    def __del__(self):
        os.remove(self.model_path)
        os.remove(self.conf_path)


class Flair(VerticalTaggerInterface):
    """Interface for Flair."""

    def __init__(self, train_path: str, dev_path: str, test_path: str) -> None:
        self.model_dir = tempfile.mkdtemp(prefix="flair_", suffix=".mod")
        train_path = Flair.convert_to_train(train_path)
        dev_path = Flair.convert_to_train(dev_path)
        test_path = Flair.convert_to_train(test_path)
        with LogTime(f"Train {train_path}"):
            subprocess.check_call(
                (
                    "nice",
                    "bin/flair-train.py",
                    "--require_gpu",
                    train_path,
                    dev_path,
                    test_path,
                    self.model_dir,
                ),
                stderr=STDERR,
            )

    def predict(self, input_path: str) -> None:
        input_path = Flair.convert_to_predict(input_path)
        temp_path = Flair.temp_path(
            input_path, self.__class__.__name__, self.ifmt
        )
        # We need to compute the final path to the model; flair-train.py makes
        #  a directory with the actual model, plus a few other random training
        #  artifacts, and that directory is what is pointed to by
        # self.model_path.
        #
        # The actual model is in that directory, named final-model.pt.
        final_model_path = os.path.join(self.model_dir, "final-model.pt")
        with open(temp_path, "w") as sink:
            with LogTime(f"Predict {input_path}"):
                subprocess.check_call(
                    (
                        "nice",
                        "bin/flair-predict.py",
                        "--require_gpu",
                        final_model_path,
                        input_path,
                    ),
                    stdout=sink,
                    stderr=STDERR,
                )
        Flair.convert_from(temp_path)

    def __del__(self):
        shutil.rmtree(self.model_dir)


def main(args):
    train = os.path.join(args.data_path, "train.textproto")
    dev = os.path.join(args.data_path, "dev.textproto")
    test = os.path.join(args.data_path, "test.textproto")
    if args.tagger == "TnT":
        tagger = TnT(train, dev, test)
        tagger.predict(dev)
        tagger.predict(test)
    elif args.tagger == "Collins":
        tagger = Collins(train, dev, test)
        tagger.predict(dev)
        tagger.predict(test)
    elif args.tagger == "LAPOS":
        tagger = LAPOS(train, dev, test)
        tagger.predict(dev)
        tagger.predict(test)
    elif args.tagger == "Stanford":
        tagger = Stanford(train, dev, test)
        tagger.predict(dev)
        tagger.predict(test)
    elif args.tagger == "NLP4J":
        tagger = NLP4J(train, dev, test)
        tagger.predict(dev)
        tagger.predict(test)
    elif args.tagger == "Flair":
        tagger = Flair(train, dev, test)
        tagger.predict(dev)
        tagger.predict(test)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tagger",
        choices=("TnT", "Collins", "LAPOS", "Stanford", "NLP4J", "Flair"),
        required=True,
        help="Tagger",
    )
    parser.add_argument("--data_path", required=True, help="input data path")
    main(parser.parse_args())
