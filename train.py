"""
网络的训练
"""
import os
from data_loader import data_load
from model import Generator, Discriminator
from base_parameters import base_parameters
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import torch

# 创建文件夹，没有才创建。有的话就有吧
os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
# 是否使用GUP
cuda = True if torch.cuda.is_available() else False
# 导入参数
opt = base_parameters()
# 损失函数
adversarial_loss = torch.nn.MSELoss()
# 初始化生成器和判别器
generator = Generator(input_dim=opt.input_dim, n_classes=opt.n_classes, img_shape=opt.input_shape)    # 生成器
discriminator = Discriminator(n_classes=opt.n_classes, img_shape=opt.input_shape)                     # 判别器
# 如果GPU可用的话
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
# 加载数据
dataloader = data_load(batch_size=opt.batch_size, img_size=opt.img_size)

# 优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 模型训练
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]
        # 对抗器的真实标签
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        #  训练生成器
        optimizer_G.zero_grad()
        # 噪声样本和对应的标签
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))     # 噪声样本
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))  # 对应的标签
        # 利用噪声样本和标签生成一系列的图片  ---->  该图片由生成器生成
        gen_imgs = generator(z, gen_labels)   # 生成器利用噪声和对应的标签生成
        # Loss measures generator's ability to fool the discriminator    # 利用判别器计算生成器生成的图片与图片对应标签的损失。
                                                                         # 生成器的目的就是根据标签和干扰生成对应的标签图片，
                                                                         # 假设判别器市准的，则目的就是希望其与真实标签1的损失最小
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)
        g_loss.backward()
        optimizer_G.step()
        #  训练判别器
        optimizer_D.zero_grad()
        # Loss for real images  真实图片的损失，目的就是希望其与真实标签1的损失最小
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)
        # Loss for fake images  生成的虚假图片的损失，目的就是希望其与虚假标签0的损失最小
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)
        # Total discriminator loss  判别器的总的损失，真实图片与虚假图片各取一般
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        # 打印每个step后的损失结果
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
        # 统计总共训练的step，每经过opt.sample_interval个step就利用当前的生成器参数进行随机生成并保存结果
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # 随机生成100个图片并显示
            z = Variable(FloatTensor(np.random.normal(0, 1, (10 ** 2, opt.latent_dim))))
            # Get labels ranging from 0 to n_classes for n rows
            labels = np.array([num for _ in range(10) for num in range(10)])
            labels = Variable(LongTensor(labels))
            gen_imgs = generator(z, labels)
            save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=10, normalize=True)
    # 保存最近一次epoch的网络权重模型到指定路径下
    torch.save(generator.state_dict(), "saved_models/generator_best.pth")