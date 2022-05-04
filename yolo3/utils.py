"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    # 先生成一个用“绝对灰”R128-G128-B128“填充的416x416新图片，然后用按比例缩放（采样方法：BICUBIC）后的输入图片粘贴，粘贴不到的部分保留为灰色
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
    new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))
    resized_image = image.resize((new_w,new_h), Image.BICUBIC)

    boxed_image = Image.new('RGB', size, (128,128,128))
    boxed_image.paste(resized_image, ((w-new_w)//2,(h-new_h)//2))
    return boxed_image
