from PIL import Image
import glob
# import tensorflow as tf
import numpy as np
import click
import json


def images_to_sprite(data):
    """
    Creates the sprite image along with any necessary padding
    Source : https://github.com/tensorflow/tensorflow/issues/6322
    Args:
      data: NxHxW[x3] tensor containing the images.
    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


def populate_img_arr(images_paths):
    """
    Get an array of images for a list of image paths
    Args:
        size: the size of image , in pixels
        should_preprocess: if the images should be processed (according to InceptionV3 requirements)
    Returns:
        arr: An array of the loaded images
    """
    arr = []
    for i, img_path in enumerate(images_paths):
        img = Image.open(img_path)
        x = np.array(img)
        arr.append(x)
    arr = np.array(arr)
    return arr


@click.command()
@click.option('--data', help='Data folder,has to end with /')
@click.option('--sprite_size', default=128, help='Size of sprite')
@click.option('--sprite_name', default="sprites.png", help='Name of sprites file')
def main(data, sprite_size, sprite_name):
    if not data.endswith('/'):
        raise ValueError('Makesure --name ends with a "/"')
    
    images_paths = glob.glob(data + "*.jpg")
    images_paths.extend(glob.glob(data + "*.JPG"))
    images_paths.extend(glob.glob(data + "*.png"))

    raw_imgs = populate_img_arr(sorted(images_paths))
    sprite = Image.fromarray(images_to_sprite(raw_imgs).astype(np.uint8))
    sprite.save(sprite_name)

if __name__ == '__main__':
    main()
