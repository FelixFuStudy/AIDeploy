#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
import PIL
import numpy as np


def get_palette(num_classes):
    """
    功能：获得画板颜色
    """
    n = num_classes
    palette = [0]*(n*3)
    for j in range(0, n):
            lab = j
            palette[j*3+0] = 0
            palette[j*3+1] = 0
            palette[j*3+2] = 0
            i = 0
            while (lab > 0):
                    palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return palette


def colorize_mask(mask, palette):
    """
    功能：为mask图，填充颜色
    """
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask
