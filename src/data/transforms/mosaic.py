"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import random

import torch
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image

from ...core import register
from .._misc import convert_to_tv_tensor

torchvision.disable_beta_transforms_warning()


@register()
class Mosaic(T.Transform):
    def __init__(
        self,
        size,
        max_size=None,
        p=1.0,
        output_size=None,
        crop=True,
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.5, 1.5),
        shear=None,
        fill=114,
    ) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Mosaic probability must be in [0, 1], got {p}.")
        self.p = p
        self.resize = T.Resize(size=size, max_size=max_size)
        self.output_size = output_size or max_size or size
        self.random_crop = T.RandomCrop(size=self.output_size) if crop else None

        self.random_affine = T.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            fill=fill,
        )

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        image, target, dataset = inputs
        if torch.rand(1) >= self.p:
            return image, target, dataset

        image, target = self.resize(image, target)
        images = [image]
        targets = [target]
        indices = random.choices(range(len(dataset)), k=3)
        for i in indices:
            image, target = dataset.load_item(i)
            image, target = self.resize(image, target)
            images.append(image)
            targets.append(target)

        h, w = F.get_size(images[0])
        offset = [[0, 0], [w, 0], [0, h], [w, h]]
        image = Image.new(mode=images[0].mode, size=(w * 2, h * 2), color=0)
        for i, im in enumerate(images):
            image.paste(im, offset[i])

        offset = torch.tensor([[0, 0], [w, 0], [0, h], [w, h]]).repeat(1, 2)
        target = {}
        for k in targets[0]:
            if k == "boxes":
                v = [t[k] + offset[i] for i, t in enumerate(targets)]
            else:
                v = [t[k] for t in targets]

            if isinstance(v[0], torch.Tensor):
                v = torch.cat(v, dim=0)

            target[k] = v

        if "boxes" in target:
            # target['boxes'] = target['boxes'].clamp(0, 640 * 2 - 1)
            w, h = image.size
            target["boxes"] = convert_to_tv_tensor(
                target["boxes"], "boxes", box_format="xyxy", spatial_size=[h, w]
            )

        if "masks" in target:
            target["masks"] = convert_to_tv_tensor(target["masks"], "masks")

        image, target = self.random_affine(image, target)
        if self.random_crop is not None:
            image, target = self.random_crop(image, target)

        return image, target, dataset
