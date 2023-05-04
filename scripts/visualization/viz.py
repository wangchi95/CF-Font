import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
import click, os
import torch
import shutil

@click.command()
@click.option('--name', default="FontStyle_228", help='Name of visualisation')
@click.option('--sprite_size', default=128, help='Size of sprite')
@click.option('--feature_dir', default='feature.npy', help='codebook npy filepath')
@click.option('--sprite', help='Name of sprites file')
@click.option('--prefix', default='viz', help='log for tensorboard')
@click.option('--suffix', default='', help='log for tensorboard')
def main(name, sprite_size, feature_dir, sprite, prefix, suffix):
    config = projector.ProjectorConfig()

    for feature, feature_suf in [['style.pth', '_s'], ['c_src.pth', '_c']]:
        feature = os.path.join(feature_dir, feature)
        # assert sprite in ['rgb', 'alpha']
        if feature.endswith('npy'):
            codebook = feature
            codebook = np.load(codebook)
        else:
            codebook = torch.load(feature).numpy()
        codebook = tf.Variable(tf.convert_to_tensor(codebook, dtype=tf.float32))
        ckpt = tf.train.Checkpoint(embedding=codebook)
        logdir = os.path.join('viz', prefix + '_' + name)
        ckpt.save(os.path.join(logdir, f'embedding{suffix}{feature_suf}.ckpt'))
    

        # You can add multiple embeddings. Here we add only one.
        embedding = config.embeddings.add()
        embedding.tensor_name = "embedding{suffix}{feature_suf}/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.sprite.image_path = os.path.join(os.getcwd(), sprite)
        embedding.sprite.single_image_dim.extend([sprite_size, sprite_size])


    projector.visualize_embeddings(logdir, config)

if __name__ == '__main__':
    main()
