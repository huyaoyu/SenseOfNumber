
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser(description="Train a simple fully connected neural network.")
    
parser.add_argument("--load-model", action="store_true", default=False, \
    help="Load saved.pt.")

parser.add_argument("--epoches", type=int, default=1000, \
    help="The number of outer looping epoches.")

parser.add_argument("--job-mode", type=str, default="train", \
    help="Set job mode. Choose one from \"train\", \"test\", \"test_v\", and \"infer\"")

parser.add_argument("--test-v-idx", type=int, default=0, \
    help="The index of test data when --job-mode is \"test_v\"")

parser.add_argument("--fn-vec", type=str, default="vectors.dat", \
    help="The file name of vectors.")

parser.add_argument("--fn-dist", type=str, default="distances.dat", \
    help="The file name of distances.")

args = parser.parse_args()
