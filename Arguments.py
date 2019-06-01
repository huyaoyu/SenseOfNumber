
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser(description="Train a simple fully connected neural network.")
    
parser.add_argument("--load-model", action="store_true", default=False, \
    help="Load saved.pt.")

parser.add_argument("--epoches", type=int, default=1000, \
    help="The number of outer looping epoches.")

parser.add_argument("--job-mode", type=str, default="train", \
    help="Set job mode. Choose one from \"train\", \"test\" and \"infer\"")

args = parser.parse_args()
