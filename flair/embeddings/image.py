import logging
from typing import List

import torch
import torch.nn.functional as F
from torch.nn import (
    AdaptiveAvgPool2d,
    AdaptiveMaxPool2d,
    Conv2d,
    Dropout2d,
    Linear,
    MaxPool2d,
    Parameter,
    ReLU,
    Sequential,
    TransformerEncoder,
    TransformerEncoderLayer,
)

import flair
from flair.data import Image
from flair.embeddings.base import Embeddings

log = logging.getLogger("flair")


class ImageEmbeddings(Embeddings[Image]):
    @property
    def embedding_type(self) -> str:
        return "image-level"


class IdentityImageEmbeddings(ImageEmbeddings):
    def __init__(self, transforms):
        import PIL as pythonimagelib

        self.PIL = pythonimagelib
        self.name = "Identity"
        self.transforms = transforms
        self.__embedding_length = None
        self.static_embeddings = True
        super().__init__()

    def _add_embeddings_internal(self, images: List[Image]):
        for image in images:
            image_data = self.PIL.Image.open(image.imageURL)
            image_data.load()
            image.set_embedding(self.name, self.transforms(image_data))

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def __str__(self):
        return self.name


class PrecomputedImageEmbeddings(ImageEmbeddings):
    def __init__(self, url2tensor_dict, name):
        self.url2tensor_dict = url2tensor_dict
        self.name = name
        self.__embedding_length = len(list(self.url2tensor_dict.values())[0])
        self.static_embeddings = True
        super().__init__()

    def _add_embeddings_internal(self, images: List[Image]):
        for image in images:
            if image.imageURL in self.url2tensor_dict:
                image.set_embedding(self.name, self.url2tensor_dict[image.imageURL])
            else:
                image.set_embedding(self.name, torch.zeros(self.__embedding_length, device=flair.device))

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def __str__(self):
        return self.name


class NetworkImageEmbeddings(ImageEmbeddings):
    def __init__(self, name, pretrained=True, transforms=None):
        super().__init__()

        try:
            import torchvision as torchvision
        except ModuleNotFoundError:
            log.warning("-" * 100)
            log.warning('ATTENTION! The library "torchvision" is not installed!')
            log.warning('To use convnets pretraned on ImageNet, please first install with "pip install torchvision"')
            log.warning("-" * 100)
            pass

        model_info = {
            "resnet50": (torchvision.models.resnet50, lambda x: list(x)[:-1], 2048),
            "mobilenet_v2": (
                torchvision.models.mobilenet_v2,
                lambda x: list(x)[:-1] + [torch.nn.AdaptiveAvgPool2d((1, 1))],
                1280,
            ),
        }

        transforms = [] if transforms is None else transforms
        transforms += [torchvision.transforms.ToTensor()]
        if pretrained:
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]
            transforms += [torchvision.transforms.Normalize(mean=imagenet_mean, std=imagenet_std)]
        self.transforms = torchvision.transforms.Compose(transforms)

        if name in model_info:
            model_constructor = model_info[name][0]
            model_features = model_info[name][1]
            embedding_length = model_info[name][2]

            net = model_constructor(pretrained=pretrained)
            modules = model_features(net.children())
            self.features = torch.nn.Sequential(*modules)

            self.__embedding_length = embedding_length

            self.name = name
        else:
            raise Exception(f"Image embeddings {name} not available.")

    def _add_embeddings_internal(self, images: List[Image]):
        image_tensor = torch.stack([self.transforms(image.data) for image in images])
        image_embeddings = self.features(image_tensor)
        image_embeddings = (
            image_embeddings.view(image_embeddings.shape[:2]) if image_embeddings.dim() == 4 else image_embeddings
        )
        if image_embeddings.dim() != 2:
            raise Exception(f"Unknown embedding shape of length {image_embeddings.dim()}")
        for image_id, image in enumerate(images):
            image.set_embedding(self.name, image_embeddings[image_id])

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def __str__(self):
        return self.name


class ConvTransformNetworkImageEmbeddings(ImageEmbeddings):
    def __init__(self, feats_in, convnet_parms, posnet_parms, transformer_parms):
        super(ConvTransformNetworkImageEmbeddings, self).__init__()

        adaptive_pool_func_map = {"max": AdaptiveMaxPool2d, "avg": AdaptiveAvgPool2d}

        convnet_arch = [] if convnet_parms["dropout"][0] <= 0 else [Dropout2d(convnet_parms["dropout"][0])]
        convnet_arch.extend(
            [
                Conv2d(
                    in_channels=feats_in,
                    out_channels=convnet_parms["n_feats_out"][0],
                    kernel_size=convnet_parms["kernel_sizes"][0],
                    padding=convnet_parms["kernel_sizes"][0][0] // 2,
                    stride=convnet_parms["strides"][0],
                    groups=convnet_parms["groups"][0],
                ),
                ReLU(),
            ]
        )
        if "0" in convnet_parms["pool_layers_map"]:
            convnet_arch.append(MaxPool2d(kernel_size=convnet_parms["pool_layers_map"]["0"]))
        for layer_id, (kernel_size, n_in, n_out, groups, stride, dropout) in enumerate(
            zip(
                convnet_parms["kernel_sizes"][1:],
                convnet_parms["n_feats_out"][:-1],
                convnet_parms["n_feats_out"][1:],
                convnet_parms["groups"][1:],
                convnet_parms["strides"][1:],
                convnet_parms["dropout"][1:],
            )
        ):
            if dropout > 0:
                convnet_arch.append(Dropout2d(dropout))
            convnet_arch.append(
                Conv2d(
                    in_channels=n_in,
                    out_channels=n_out,
                    kernel_size=kernel_size,
                    padding=kernel_size[0] // 2,
                    stride=stride,
                    groups=groups,
                )
            )
            convnet_arch.append(ReLU())
            if str(layer_id + 1) in convnet_parms["pool_layers_map"]:
                convnet_arch.append(MaxPool2d(kernel_size=convnet_parms["pool_layers_map"][str(layer_id + 1)]))
        convnet_arch.append(
            adaptive_pool_func_map[convnet_parms["adaptive_pool_func"]](output_size=convnet_parms["output_size"])
        )
        self.conv_features = Sequential(*convnet_arch)
        conv_feat_dim = convnet_parms["n_feats_out"][-1]
        if posnet_parms is not None and transformer_parms is not None:
            self.use_transformer = True
            if posnet_parms["nonlinear"]:
                posnet_arch = [
                    Linear(2, posnet_parms["n_hidden"]),
                    ReLU(),
                    Linear(posnet_parms["n_hidden"], conv_feat_dim),
                ]
            else:
                posnet_arch = [Linear(2, conv_feat_dim)]
            self.position_features = Sequential(*posnet_arch)
            transformer_layer = TransformerEncoderLayer(
                d_model=conv_feat_dim, **transformer_parms["transformer_encoder_parms"]
            )
            self.transformer = TransformerEncoder(transformer_layer, num_layers=transformer_parms["n_blocks"])
            # <cls> token initially set to 1/D, so it attends to all image features equally
            self.cls_token = Parameter(torch.ones(conv_feat_dim, 1) / conv_feat_dim)
            self._feat_dim = conv_feat_dim
        else:
            self.use_transformer = False
            self._feat_dim = convnet_parms["output_size"][0] * convnet_parms["output_size"][1] * conv_feat_dim

    def forward(self, x):
        x = self.conv_features(x)  # [b, d, h, w]
        b, d, h, w = x.shape
        if self.use_transformer:
            # add positional encodings
            y = torch.stack(
                [
                    torch.cat([torch.arange(h).unsqueeze(1)] * w, dim=1),
                    torch.cat([torch.arange(w).unsqueeze(0)] * h, dim=0),
                ]
            )  # [2, h, w
            y = y.view([2, h * w]).transpose(1, 0)  # [h*w, 2]
            y = y.type(torch.float32).to(flair.device)
            y = self.position_features(y).transpose(1, 0).view([d, h, w])  # [h*w, d] => [d, h, w]
            y = y.unsqueeze(dim=0)  # [1, d, h, w]
            x = x + y  # [b, d, h, w] + [1, d, h, w] => [b, d, h, w]
            # reshape the pixels into the sequence
            x = x.view([b, d, h * w])  # [b, d, h*w]
            # layer norm after convolution and positional encodings
            x = F.layer_norm(x.permute([0, 2, 1]), (d,)).permute([0, 2, 1])
            # add <cls> token
            x = torch.cat([x, torch.stack([self.cls_token] * b)], dim=2)  # [b, d, h*w+1]
            # transformer requires input in the shape [h*w+1, b, d]
            x = (
                x.view([b * d, h * w + 1]).transpose(1, 0).view([h * w + 1, b, d])
            )  # [b, d, h*w+1] => [b*d, h*w+1] => [h*w+1, b*d] => [h*w+1, b*d]
            x = self.transformer(x)  # [h*w+1, b, d]
            # the output is an embedding of <cls> token
            x = x[-1, :, :]  # [b, d]
        else:
            x = x.view([-1, self._feat_dim])
            x = F.layer_norm(x, (self._feat_dim,))

        return x

    def _add_embeddings_internal(self, images: List[Image]):
        image_tensor = torch.stack([image.data for image in images])
        image_embeddings = self.forward(image_tensor)
        for image_id, image in enumerate(images):
            image.set_embedding(self.name, image_embeddings[image_id])

    @property
    def embedding_length(self):
        return self._feat_dim

    def __str__(self):
        return self.name
