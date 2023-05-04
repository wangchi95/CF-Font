from PIL import Image,ImageDraw,ImageFont
import os
import numpy as np
import pathlib
import argparse


parser = argparse.ArgumentParser(description='Obtaining characters from .ttf')
parser.add_argument('--ttf_path', type=str, default='../ttf_folder',help='ttf directory')
parser.add_argument('--chara', type=str, default='../chara.txt',help='characters')
parser.add_argument('--save_path', type=str, default='../save_folder',help='images directory')
parser.add_argument('--img_size', type=int, help='The size of generated images')
parser.add_argument('--chara_size', type=int, help='The size of generated characters')
parser.add_argument('--start_id', type=int, default=0, help='The start index for save')
parser.add_argument('--only_id', type=int, default=-1, help='The only index for save')
args = parser.parse_args()

file_object = open(args.chara,encoding='utf-8')   
try:
	characters = file_object.read()
finally:
    file_object.close()


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    return img

def draw_example(ch, src_font, canvas_size, x_offset, y_offset):
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    example_img.paste(src_img, (0, 0))
    return example_img

data_dir = args.ttf_path
data_root = pathlib.Path(data_dir)
# print(data_root)

all_image_paths = list(data_root.glob('*.tt*')) + list(data_root.glob('*.TT*')) + list(data_root.glob('*.ot*')) + list(data_root.glob('*.OT*'))
all_image_paths = [str(path) for path in all_image_paths]
all_image_paths = sorted(all_image_paths)
all_image_paths = all_image_paths[args.start_id:]

seq = list()
# Auto Run
print(len(all_image_paths))
for (label,item) in zip(range(args.start_id, args.start_id+len(all_image_paths)),all_image_paths):
    print(label, item)
    if args.only_id == -1 or args.only_id == label:
        lrs = []
        tds = []
        # for sample_i in range(100):
        for sample_i in range(len(characters)):
            src_font = ImageFont.truetype(item, size = args.chara_size)
            try:
                # check pos
                # mean 
                chara_base = characters[sample_i]
                img_base = 255 - np.array(draw_example(chara_base, src_font, args.img_size, (args.img_size-args.chara_size)/2, (args.img_size-args.chara_size)/2))
                img_base = img_base.sum(2)
                img_base_dim0 = img_base.sum(1)
                img_base_dim1 = img_base.sum(0)
                pos_dim0 = np.where(img_base_dim0 > 0)[0]
                pos_dim1 = np.where(img_base_dim1 > 0)[0]
                top, down = pos_dim0.min(), args.img_size - pos_dim0.max()
                left, right = pos_dim1.min(), args.img_size - pos_dim1.max()
                lr = 0 if abs(left - right) < 2 else right-left
                td = 0 if abs(top - down) < 2 else down-top
                lrs.append(lr)
                tds.append(td)
            except:
                print(f'Skip check sample {sample_i}')
        lr = np.mean(lrs) 
        td = np.mean(tds)
        
        try:
            if left + right > args.img_size * 2 / 3 or top + down > args.img_size * 2 / 3:
                print('!!!', label, item)
        except:
            lr = td = 0
        # exit()
        # characters_now = characters[:405] if label == 0 else characters
        characters_now = characters

        for (chara,cnt) in zip(characters_now, range(len(characters_now))):
            img = draw_example(chara, src_font, args.img_size, (args.img_size-args.chara_size + lr)/2, (args.img_size-args.chara_size + td)/2)
            path_full = os.path.join(args.save_path, 'id_%d'%label)
            if not os.path.exists(path_full):
                os.mkdir(path_full)
            img.save(os.path.join(path_full, "%04d.png" % (cnt)))