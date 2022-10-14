"""mASSC implementation."""

import torch
from torchmetrics.functional.classification import accuracy


def mASSC(y_img1, y_img2, offsets):
    """
    Compute the mASSC metric.

    Args:
        y_img1 (torch.Tensor): The first image.
        y_img2 (torch.Tensor): The second image.
        offsets_1 (torch.Tensor): The offsets of the first image.
        offsets_2 (torch.Tensor): The offsets of the second image.

    Returns:
        torch.Tensor: The mASSC metric.
    """
    # Compute the mASSC metric.
    _h, _w = y_img1.shape[-2:]

    if offsets[0] >= 0:
        y1_h0 = offsets[0]
        y2_h0 = 0

        y1_h1 = _h
        y2_h1 = _h - offsets[0]
    else:
        y1_h0 = 0
        y2_h0 = -offsets[0]

        y1_h1 = _h + offsets[0]
        y2_h1 = _h

    if offsets[1] >= 0:
        y1_w0 = offsets[1]
        y2_w0 = 0

        y1_w1 = _w
        y2_w1 = _w - offsets[1]
    else:
        y1_w0 = 0
        y2_w0 = -offsets[1]

        y1_w1 = _w + offsets[1]
        y2_w1 = _w

    y1 = torch.argmax(y_img1[:, :, y1_h0:y1_h1, y1_w0:y1_w1], dim=1)
    y1 = torch.flatten(y1, start_dim=1)
    y2 = torch.argmax(y_img2[:, :, y2_h0:y2_h1, y2_w0:y2_w1], dim=1)
    y2 = torch.flatten(y2, start_dim=1)

    return accuracy(y1, y2)


def mASSC_circular(y_img1, y_img2, offsets):
    """
    Compute the mASSC metric. (for circular shifts)

    Args:
        y_img1 (torch.Tensor): The first image.
        y_img2 (torch.Tensor): The second image.
        offsets_1 (torch.Tensor): The offsets of the first image.
        offsets_2 (torch.Tensor): The offsets of the second image.

    Returns:
        torch.Tensor: The mASSC metric.
    """
    # Compute the mASSC metric.
    roll_dims = (-2, -1)
    offsets = tuple(-d for d in offsets)
    y_img2 = torch.roll(y_img2, offsets, dims=roll_dims)

    y1 = torch.argmax(y_img1, dim=1)
    y1 = torch.flatten(y1, start_dim=1)

    y2 = torch.argmax(y_img2, dim=1)
    y2 = torch.flatten(y2, start_dim=1)

    return accuracy(y1, y2)
