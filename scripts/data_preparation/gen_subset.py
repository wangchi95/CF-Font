from tqdm import tqdm
import argparse
import shutil
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-ic', '--input_chara', type=str, required=True)
parser.add_argument('-oc', '--output_chara', type=str, required=True)

args = parser.parse_args()

def get_charas(path):
    with open(path,encoding='utf-8') as f:
        characters = f.read()
    return list(characters)

def safe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

ip = args.input
op = args.output
ic = get_charas(args.input_chara)
oc = get_charas(args.output_chara)

assert len(set(oc) - set(ic)) == 0
ic_mapper = {c:i for i,c in enumerate(ic)}

safe_mkdir(op)

for sub_folder in tqdm(sorted(os.listdir(ip))):
    if not os.path.isdir(os.path.join(ip, sub_folder)): continue
    safe_mkdir(os.path.join(op, sub_folder))
    for out_idx, out_char in enumerate(oc):
        in_idx = ic_mapper[out_char]
        src = os.path.join(ip, sub_folder, '%04d.png' % in_idx)
        dst = os.path.join(op, sub_folder, '%04d.png' % out_idx)
        shutil.copyfile(src, dst)