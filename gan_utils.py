import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np 

def plot_im(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def test_provider(prov):
    im =  prov.next()[0][0]
    print(im.shape)
    im = np.reshape(im, (32, 32, 3))
    plt.imshow(im)
    plt.show()
    