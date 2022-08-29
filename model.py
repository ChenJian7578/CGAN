"""
定义生成器模型和判别器模型
"""
import numpy as np
import torch.nn as nn
import torch

# 生成器模型
class Generator(nn.Module):
    def __init__(self, input_dim, n_classes, img_shape):
        """
        :param input_dim:  干扰数据的长度
        :param n_classes:  数据集包含的目标种类
        :param img_shape:  想要生成的图片的尺寸(与判别器输入的图像尺寸保持一致)
        """
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(self.n_classes, self.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn

                          .LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.input_dim + self.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # 连接标签嵌入和图像以产生输入
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *(self.img_shape))
        return img

# 判别器模型
class Discriminator(nn.Module):
    def __init__(self, n_classes, img_shape):
        """
        :param n_classes: 数据集包含的目标种类
        :param img_shape: 输入图片的尺寸
        """
        super(Discriminator, self).__init__()
        self.n_classes = n_classes
        self.img_shape = img_shape
        self.label_embedding = nn.Embedding(self.n_classes, self.n_classes)

        self.model = nn.Sequential(
            nn.Linear(self.n_classes + int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # 连接标签嵌入和图像以产生输入
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

if __name__ == "__main__":
    # 定义一些变量
    input_dim = 100           # 干扰输入
    class_nummber = 10        # 数据集类别数
    img_size = [1, 32, 32]    # 图像形状
    # 创建判别器
    d_model = Discriminator(n_classes=class_nummber, img_shape= img_size)
    # 打印判别模型
    print(d_model.model)
    print('\n')
    # 创建生成器
    g_model = Generator(input_dim=input_dim, n_classes=class_nummber, img_shape=img_size)
    # 打生成模型
    print(g_model.model)
    print('\n')








