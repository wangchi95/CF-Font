import numpy as np
import argparse
import shutil
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-b', '--basis', type=str, required=True)

args = parser.parse_args()

i = args.input
o = args.output
# basis = np.load(args.basis)
basis = np.loadtxt(args.basis, dtype='int')

if not os.path.exists(o): os.mkdir(o)

# check
print(basis)

for ido, idi in enumerate(basis):
    src = os.path.join(i, f"id_{idi}")
    dst = os.path.join(o, f"id_{ido}")
    print(src, '-->', dst)
    shutil.copytree(src, dst)
