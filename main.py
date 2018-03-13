import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from data_providers import MNISTDataProvider
import models


if __name__ == '__main__':
    dataset = MNISTDataProvider(batch_size=64)
    
    steps = {"save": 5000, "sample": 500}
    
    """
    gan = models.MNIST_GAN(
            model_name="test",
            batch_size=100,
            dataset=dataset,
            num_iter=10000,
            z_dim=64,
            im_dim=28,
            steps = steps)
    
    gan = models.MNIST_DCGAN(
            model_name="mnist_dcgan",
            batch_size=64,
            dataset=dataset,
            num_iter=10000,
            z_dim=64,
            im_dim=28,
            steps = steps)

    gan = models.WGAN_conv(
            model_name="mnist_wgan_conv",
            batch_size=64,
            dataset=dataset,
            num_iter=100000,
            z_dim=64,
            im_dim=28,
            steps = steps)
    """
    gan = models.WGAN_simple(
        model_name="mnist_wgan_simple",
        batch_size=64,
        dataset=dataset,
        num_iter=100000,
        z_dim=64,
        im_dim=28,
        steps = steps)
 
    gan.train()
    