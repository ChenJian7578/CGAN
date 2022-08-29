"""
设置的默认参数
"""
import argparse

def base_parameters():
    """
    :return: 默认参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="图像采样间隔")
    parser.add_argument("--input_shape", type=tuple, default=(1, 32, 32), help="输入图像的尺寸")
    parser.add_argument("--input_dim", type=int, default=100, help="生成器输出参数的长度")
    parser.add_argument("--class_nummber", type=int, default=10, help="数据集中的类别数")
    opt = parser.parse_args()
    return opt







