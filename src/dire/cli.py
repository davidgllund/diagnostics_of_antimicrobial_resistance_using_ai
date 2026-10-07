import argparse
import sys

from scripts import evaluate_model as evaluate_module
from scripts import fine_tune_model as fine_tune_module
from scripts import process_data as process_data_module
from scripts import train_model as train_module


def _invoke(module, argv):
    module.argv = list(argv)
    return module.main()


def train(argv=None):
    args = sys.argv[2:] if argv is None else list(argv)
    return _invoke(train_module, args)


def evaluate(argv=None):
    args = sys.argv[2:] if argv is None else list(argv)
    return _invoke(evaluate_module, args)


def fine_tune(argv=None):
    args = sys.argv[2:] if argv is None else list(argv)
    return _invoke(fine_tune_module, args)


def process_data(argv=None):
    args = sys.argv[2:] if argv is None else list(argv)
    return _invoke(process_data_module, args)


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    if not args:
        parser = argparse.ArgumentParser(prog="dire")
        parser.print_help()
        return 0

    command = args[0]
    command_args = args[1:]

    if command == "train":
        return train(command_args)
    if command == "evaluate":
        return evaluate(command_args)
    if command == "fine-tune":
        return fine_tune(command_args)
    if command == "process-data":
        return process_data(command_args)

    parser = argparse.ArgumentParser(prog="dire")
    parser.add_argument("command", choices=["train", "evaluate", "fine-tune", "process-data"])
    parser.print_help()
    return 1
