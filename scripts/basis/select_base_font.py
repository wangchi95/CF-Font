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
basis = np.load(args.basis)

if not os.path.exists(o): os.mkdir(o)
fns = sorted(os.listdir(i))
valid_ends = ['ttf', 'otf', 'ttc']
checker = lambda i: any([i.lower().endswith(end) for end in valid_ends])
fns = np.array([fn for fn in fns if checker(fn)])

print(len(fns))
print(basis)
print(fns[basis])

for fn in fns[basis]:
    shutil.copyfile(os.path.join(i, fn), os.path.join(o, fn))