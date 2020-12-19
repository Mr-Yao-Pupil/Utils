from PIL import Image


def image_proportional_scaling(image, max_size):
    """
    图片等比缩放
    :param image: PIL打开的图片实例对象
    :param max_size: 缩放后最大边的长度
    :return: 等比缩放后的Image对象
    """
    w, h = image.max_size
    new_ar = w / h
    if new_ar < 1:
        new_h = max_size
        new_w = int((max_size / h) * w)
    else:
        new_w = max_size
        new_h = int((max_size / w) * h)
    return image.resize((new_w, new_h), Image.BICUBIC)


def image_pad_to_square(image, pad_color=0):
    """
    将图片填充成正方形，不改变图片最大边大小
    :param image: PIL打开的图片实例对象
    :param pad_color: 填充的颜色
    :return: 补边后的Image对象
    """
    if not isinstance(pad_color, int) or isinstance(pad_color, tuple) or isinstance(pad_color, list):
        raise ValueError(f"pad_color传入参数错误,应当传入int, float, tuple,当前传入类型为:{type(pad_color)}")
    if not ''.join(image.getbands()) in ['1', 'L', 'P', 'RGB', 'RGBA', 'CMYK']:
        raise TypeError(f"不支持格式{''.join(image.getbands())}")
    w, h = image.size
    mode = ''.join(image.getbands())
    max_side = max(w, h)
    if isinstance(pad_color, int):
        new_image = Image.new(mode, size=(max_side, max_side),
                              color=tuple([pad_color] * len(mode)))
    elif isinstance(pad_color, tuple) or isinstance(pad_color, list):
        if len(mode) == len(pad_color):
            new_image = Image.new(mode, size=(max_side, max_side), color=pad_color)
        else:
            raise ValueError(f"输入tuple或着list的长度与图片通道数不符")
    else:
        raise ValueError(f"未知错误")

    if max_side == w:
        dx = 0
        dy = int((max_side - h) / 2)
    else:
        dx = int((max_side - w) / 2)
        dy = 0
    new_image.paste(image, (dx, dy))
    return new_image
