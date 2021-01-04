import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import os


class SaveConvFeatures():

    def __init__(self, m):  # module to hook
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.data

    def remove(self):
        self.hook.remove()


def show_feature_map(image_path, conv_features, issave=False, save_path=None, isshow=False):
    """可视化卷积层特征图输出
    :param image_path:源图像文件路径
    :param conv_features:得到的卷积输出,[b, c, h, w]
    :param issave: 是否存储热力图, bool
    :param save_path:存储地址
    :param isshow:是否plt展示热力图
    """
    img = Image.open(image_path).convert('RGB')
    heat = conv_features.squeeze(0)  # 降维操作,尺寸变为(2048,7,7)
    heat_mean = torch.mean(heat.cpu(), dim=0)  # 对各卷积层(2048)求平均值,尺寸变为(7,7)
    heatmap = heat_mean.numpy()  # 转换为numpy数组
    heatmap /= np.max(heatmap)  # minmax归一化处理
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))  # 变换heatmap图像尺寸,使之与原图匹配,方便后续可视化
    heatmap = np.uint8(255 * heatmap)  # 像素值缩放至(0,255)之间,uint8类型,这也是前面需要做归一化的原因,否则像素值会溢出255(也就是8位颜色通道)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 颜色变换
    plt.imshow(heatmap)
    plt.show()
    # heatmap = np.array(Image.fromarray(heatmap).convert('L'))
    superimg = heatmap * 0.4 + np.array(img)[:, :, ::-1]  # 图像叠加，注意翻转通道，cv用的是bgr

    if issave:
        if not isinstance(save_path, str):
            raise TypeError(f"图片存储地址类型错误，应该传入字符变量，而不是{type(save_path)}")
        cv2.imwrite(save_path, superimg)  # 保存结果
    # 可视化叠加至源图像的结果
    if isshow:
        if issave:
            img_ = np.array(Image.open(save_path).convert("RGB"))
            plt.imshow(img_)
            plt.show()
        else:
            cv2.imwrite('cache.jpg', superimg)
            img_ = np.array(Image.open('cache.jpg').convert("RGB"))
            plt.imshow(img_)
            plt.show()
            os.remove('cache.jpg')
            # print(superimg.shape)


def draw_cam(model, image_path, trans, model_layer, device='cuda', issave=False, save_path=None, isshow=False):
    """
    绘制热力图
    :param model: 使用的模型, nn.Model
    :param image_path: 图片地址
    :param trans: 图片读入后转化tensor的方式
    :param model_layer: 绘制目标网络层的热力图
    :param device: 设备名称
    :return: None
    """
    if not isinstance(model, torch.nn.Module):
        raise ValueError(f"模型传入错误, 仅能传入nn.Model或其子类, 不支持{type(model)}")
    if not isinstance(image_path, str):
        raise ValueError(f"图片地址数据类型错误, 仅能传入字符串类型, 不支持{type(image_path)}")
    if not isinstance(trans, transforms.Compose):
        raise ValueError(f"图片转换类型错误, 仅能传入transforms.Compos的实例对象, 不能传入{type(trans)}")
    assert device == "cuda" or device == 'cpu', "device数据类型错误，仅能传入'cuda'或者'cpu'"

    cudnn.bevhmark = True

    hook_ref = SaveConvFeatures(model_layer)

    image = Image.open(image_path).convert("RGB")
    image = trans(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        model(image)

    conv_features = hook_ref.features

    show_feature_map(image_path, conv_features, issave=issave, save_path=save_path, isshow=isshow)
