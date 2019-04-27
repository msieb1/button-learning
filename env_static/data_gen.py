import os, argparse
import numpy as np
import matplotlib.pyplot as plt

from autolab_core import YamlConfig

from button import Button, COLORS, SHAPES

def imshow(im, save=None):
    im = im / np.max(np.abs(im))
    plt.figure()
    if len(im.shape) == 2:
        if save:
            plt.imsave(save, im, cmap='gray')
        else:
            plt.imshow(im, cmap='gray')
    else:
        if save:
            plt.imsave(save, im)
        else:
            plt.imshow(im)
    if not save:
        plt.show()
    plt.clf()
    plt.cla()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', '-o', type=str)
    parser.add_argument('--num', '-n', type=int, default=100)
    parser.add_argument('--cfg', '-c', type=str, default='data_gen.yaml')
    args = parser.parse_args()

    if os.path.isdir(args.logdir):
        raise ValueError('Cannot overwrite data!')
    os.makedirs(args.logdir)

    cfg = YamlConfig(args.cfg)

    bs = Button.sample(args.num)
    colors, shapes, is_buttons = [], [], []
    for i, b in enumerate(bs):
        colors.append(b.color)
        shapes.append(b.shape)

        is_button = 1 if cfg['is_button']['color'][COLORS.get_name(b.color)] \
                and cfg['is_button']['shape'][SHAPES.get_name(b.shape)] else 0
        is_buttons.append(is_button)

        imshow(b.render(im_size=cfg['im_size']), save=os.path.join(args.logdir, '{0:06d}.png'.format(i)))

    np.savez(os.path.join(args.logdir, 'data.npz'), colors=colors, shapes=shapes, is_buttons=is_buttons)
    import IPython; IPython.embed(); exit(0)