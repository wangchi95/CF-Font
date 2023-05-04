import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--full', type=str, required=True, help='Full chars')
parser.add_argument('-t', '--train', type=str, required=True, help='Train chars')
parser.add_argument('-o', '--out', type=str, required=True, help='save chars')
args = parser.parse_args()

with open(args.full, 'r') as f:
    chars_f = f.readline()
with open(args.train, 'r') as f:
    chars_t = f.readline()

chars_f = set(chars_f)
chars_t = set(chars_t)
chars_o = sorted(list(chars_f - chars_t))
print(chars_o, len(chars_o))

with open(args.out, 'w') as f:
    f.write(''.join(chars_o))