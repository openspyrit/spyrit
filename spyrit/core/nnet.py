"""
Neural network models for image denoising.
"""

# from __future__ import print_function, division
import torch
import torch.nn as nn
from collections import OrderedDict
import copy


# =============================================================================
class Unet(nn.Module):
    """Defines a U-Net model.

    This U-Net model is mostly used for image denoising, but can be used for
    other image-to-image tasks.

    The model is composed of a descending branch
    and an ascending branch. The descending branch is composed of three
    convolutional blocks, each followed by a max pooling layer. The bottleneck
    is a convolutional block with two convolutional layers. The ascending
    branch is composed of three convolutional blocks, each followed by an
    upsampling layer. The final layer is a convolutional block with two
    convolutional layers.

    The upsampling layer can be either a transposed convolution or an upsampling
    followed by a convolution. The upsampling method can be specified using the
    `upsample` and `upsample_mode` arguments.

    Args:
        in_channel (int): Number of input channels.

        out_channel (int): Number of output channels.

        upsample (bool): If True, use an upsampling layer followed by a
        convolution layer in the ascending branch instead of a transposed convolution.

        upsample_mode (str): The upsampling method to use. It is directly passed
        to the `mode` argument of the `torch.nn.Upsample` class. It can be either
        'nearest', 'bilinear', or 'bicubic'.

    Attributes:
        upsample (bool): If True, use an upsampling layer followed by a
        convolution layer in the ascending branch instead of a transposed convolution.

        upsample_mode (str): The upsampling method to use. It is directly passed
        to the `mode` argument of the `torch.nn.Upsample` class. It can be either
        'nearest', 'bilinear', or 'bicubic'.

        conv_encode1 (torch.nn.Sequential): The first convolutional block of the
        descending branch.

        conv_maxpool1 (torch.nn.MaxPool2d): The first max pooling layer.

        conv_encode2 (torch.nn.Sequential): The second convolutional block of the
        descending branch.

        conv_maxpool2 (torch.nn.MaxPool2d): The second max pooling layer.

        conv_encode3 (torch.nn.Sequential): The third convolutional block of the
        descending branch.

        conv_maxpool3 (torch.nn.MaxPool2d): The third max pooling layer.

        bottleneck (torch.nn.Sequential): The bottleneck block.

        conv_decode4 (torch.nn.Sequential): The first convolutional block of the
        ascending branch.

        conv_decode3 (torch.nn.Sequential): The second convolutional block of the
        ascending branch.

        conv_decode2 (torch.nn.Sequential): The third convolutional block of the
        ascending branch.

        final_layer (torch.nn.Sequential): The final convolutional block.

    Example:
        >>> model = Unet(in_channel=1, out_channel=1, upsample=True, upsample_mode='nearest')
        >>> x = torch.randn(1, 1, 256, 256)
        >>> y = model(x)
    """

    def __init__(
        self,
        in_channel: int = 1,
        out_channel: int = 1,
        upsample: bool = False,
        upsample_mode: str = "nearest",
    ):
        super(Unet, self).__init__()
        self.upsample = upsample
        self.upsample_mode = upsample_mode
        # Descending branch
        self.conv_encode1 = self.contract(in_channels=in_channel, out_channels=16)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contract(16, 32)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contract(32, 64)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.bottleneck = self.bottle_neck(64)
        # Decode branch
        self.conv_decode4 = self.expans(64, 64, 64)
        self.conv_decode3 = self.expans(128, 64, 32)
        self.conv_decode2 = self.expans(64, 32, 16)
        self.final_layer = self.final_block(32, 16, out_channel)

    def contract(
        self, in_channels: int, out_channels: int, kernel_size=3, padding=1
    ) -> torch.nn.Sequential:
        """Defines a convolutional block.

        It is composed of two convolutional layers followed by a ReLU activation
        function and a batch normalization layer.

        Args:
            in_channels (int): Number of channels in the input tensor.

            out_channels (int): Number of channels in the output tensor.

            kernel_size (int or tuple, optional): Size of the kernel in the convolution
            layers. It is passed to :class:`torch.nn.Conv2d`. Defaults to 3.

            padding (int or tuple or string, optional): Input padding. It is
            directly passed to :class:`torch.nn.Conv2d`, see its documentation
            for valid options. Defaults to 1.

        Returns:
            torch.nn.Sequential: A sequential block with two convolutional layers,
            ReLU activation function, and batch normalization layer.
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                padding=padding,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=out_channels,
                out_channels=out_channels,
                padding=padding,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def expans(
        self,
        in_channels: int,
        mid_channel: int,
        out_channels: int,
        kernel_size=3,
        padding=1,
    ) -> torch.nn.Sequential:
        """Defines an upsampling block.

        It is composed of two convolutional layers followed by a ReLU activation
        function and a batch normalization layer. The block finally contains
        an upsampling layer. The upsampling layer can be either a transposed
        convolution or an upsampling followed by a convolution, depending on the
        value of the `upsample` attribute defined at initialization.

        Args:
            in_channels (int): Number of channels in the input tensor.

            mid_channel (int): Number of channels in the intermediate tensor (i.e.)
            the tensor after the first convolutional layer.

            out_channels (int): Number of channels in the output tensor.

            kernel_size (int or tuple, optional): Size of the two convolution
            kernels. If the upsampling layer is a upsampling followed by a
            convolution, this kernel size is also used in that convolution. This
            argument is directly passed to :class:`torch.nn.Conv2d`. Defaults to 3.

            padding (int or tuple or string, optional): Input padding. It is
            directly passed to :class:`torch.nn.Conv2d`, see its documentation
            for valid options. Defaults to 1.

        Returns:
            torch.nn.Sequential: A sequential block with two convolutional layers,
            ReLU activation function, batch normalization layer, and an upsampling
            layer.
        """
        if self.upsample:
            upsample_subblock = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode=self.upsample_mode),
                torch.nn.Conv2d(
                    kernel_size=kernel_size,
                    in_channels=mid_channel,
                    out_channels=out_channels,
                    padding=padding,
                ),
            )
        else:
            upsample_subblock = torch.nn.ConvTranspose2d(
                kernel_size=kernel_size,
                in_channels=mid_channel,
                out_channels=out_channels,
                stride=2,
                padding=padding,
                output_padding=1,
            )
        block = torch.nn.Sequential(
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=mid_channel,
                padding=padding,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=mid_channel,
                out_channels=mid_channel,
                padding=padding,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            upsample_subblock,
        )

        return block

    def concat(self, upsampled: torch.tensor, bypass: torch.tensor) -> torch.tensor:
        """Concatenates two tensors along the channel dimension.

        This function is useful in the ascending branch of the U-Net model.

        Args:
            upsampled (torch.tensor): The upsampled tensor, which has been processed
            by the ascending part of the U-Net model.

            bypass (torch.tensor): The tensor from the descending part of the U-Net
            model, which is concatenated with the upsampled tensor.

        Returns:
            torch.tensor: The concatenated tensor [upsampled, bypass].
        """
        return torch.cat((upsampled, bypass), 1)

    def bottle_neck(
        self, in_channels: int, kernel_size=3, padding=1
    ) -> torch.nn.Sequential:
        """Defines the bottleneck block of the U-Net model.

        The bottleneck block is composed of two convolutional layers, each followed
        by a ReLU activation function. The number of output channels of the first
        convolutional layer is twice the number of its input channels.

        Args:
            in_channels (int): Number of channels in the input tensor.

            kernel_size (int or tuple, optional): Size of the two convolution
            kernels. It is directly passed to :class:`torch.nn.Conv2d` .Defaults to 3.

            padding (int or tuple or string, optional): Input padding. It is
            directly passed to :class:`torch.nn.Conv2d`, see its documentation
            for valid options. Defaults to 1.

        Returns:
            torch.nn.Sequential: A sequential block with two convolutional layers
            and ReLU activation functions.
        """
        bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=2 * in_channels,
                padding=padding,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=2 * in_channels,
                out_channels=in_channels,
                padding=padding,
            ),
            torch.nn.ReLU(),
        )
        return bottleneck

    def final_block(
        self, in_channels: int, mid_channel: int, out_channels: int, kernel_size=3
    ) -> torch.nn.Sequential:
        """Defines the final block of the U-Net model.

        The final block is composed of three convolutional layers. The first two
        are followed by a ReLU activation function and a batch normalization layer.
        The last convolutional layer is only composed of a convolution operation.

        Args:
            in_channels (int): Number of channels in the input tensor.

            mid_channel (int): Number of channels in the intermediate tensor (i.e.)
            the tensor after the first and second convolutional layers.

            out_channels (int): Number of channels in the output tensor.

            kernel_size (int or tuple, optional): Size of the three convolution
            kernels. It is directly passed to :class:`torch.nn.Conv2d` .Defaults to 3.

        Returns:
            torch.nn.Sequential: A sequential block with three convolutional layers
            and two ReLU activation functions / batch normalization layers.
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=mid_channel,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=mid_channel,
                out_channels=mid_channel,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=mid_channel,
                out_channels=out_channels,
                padding=1,
            ),
        )
        return block

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass of the U-Net model.

        The number of channels in the input tensor must be equal to the number of
        channels specified at initialization.

        Args:
            x (torch.tensor): The input tensor. It is expected to have the shape
            `b, in_channel, h, w`, where `b` is the batch size, `in_channel` is
            the number of input channels, `h` is the height, and `w` is the width.

        Returns:
            torch.tensor: The output tensor of the U-Net model. It has shape
            `b, out_channel, h, w`, where `out_channel` is the number of output
            channels specified at initialization.
        """
        # Encode
        encode_block1 = self.conv_encode1(x)
        x = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(x)
        x = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(x)
        x = self.conv_maxpool3(encode_block3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode
        x = self.conv_decode4(x)
        x = self.concat(x, encode_block3)
        x = self.conv_decode3(x)
        x = self.concat(x, encode_block2)
        x = self.conv_decode2(x)
        x = self.concat(x, encode_block1)
        x = self.final_layer(x)
        return x


# =============================================================================
class ConvNet(nn.Module):
    """A simple convolutional neural network model, without batch normalization.

    This model is composed of three convolutional layers. The first two layers
    are followed by a ReLU activation function. The last layer does not have
    any activation function. There is no batch normalization layer in this model.
    To use a convolutional neural network with batch normalization, use the
    :class:`ConvNetBN` class.

    The kernel size of the first layer is 9, the second layer is 1, and the
    third layer is 5. The stride of all layers is 1, and the padding of the
    three layers are 4, 0, and 2, respectively. The number of output channels
    of the first layer is 64, the second layer is 32, and the third layer is 1.

    This class has no arguments.

    Attributes:
        convnet (torch.nn.Sequential): The convolutional neural network model.
        It contains an ordered dictionary with the following keys:
        - 'conv1': The first convolutional layer.
        - 'relu1': The ReLU activation function after the first layer.
        - 'conv2': The second convolutional layer.
        - 'relu2': The ReLU activation function after the second layer.
        - 'conv3': The third convolutional layer.
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        self.convnet = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)),
                    ("relu2", nn.ReLU()),
                    ("conv3", nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)),
                ]
            )
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass of the ConvNet model.

        Args:
            x (torch.tensor): The input tensor. It is expected to have the shape
            `b, 1, h, w`, where `b` is the batch size, `h` is the height, and
            `w` is the width.

        Returns:
            torch.tensor: The output tensor of the ConvNet model. It has shape
            `b, 1, h, w`.
        """
        return self.convnet(x)


# =============================================================================
class ConvNetBN(nn.Module):
    """A simple convolutional neural network model, with batch normalization.

    This model is composed of three convolutional layers. The first two layers
    are followed by a ReLU activation function and a batch normalization layer.
    The last layer does not have any activation function nor batch normalization.

    The kernel size of the first layer is 9, the second layer is 1, and the
    third layer is 5. The stride of all layers is 1, and the padding of the
    three layers are 4, 0, and 2, respectively. The number of output channels
    of the first layer is 64, the second layer is 32, and the third layer is 1.

    This class has no arguments.

    Attributes:
        convnet (torch.nn.Sequential): The convolutional neural network model.
        It contains an ordered dictionary with the following keys:
        - 'conv1': The first convolutional layer.
        - 'relu1': The ReLU activation function after the first layer.
        - 'BN1': The batch normalization layer after the first layer.
        - 'conv2': The second convolutional layer.
        - 'relu2': The ReLU activation function after the second layer.
        - 'BN2': The batch normalization layer after the second layer.
        - 'conv3': The third convolutional layer.
    """

    def __init__(self):
        super(ConvNetBN, self).__init__()
        self.convnet = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4)),
                    ("relu1", nn.ReLU()),
                    ("BN1", nn.BatchNorm2d(64)),
                    ("conv2", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)),
                    ("relu2", nn.ReLU()),
                    ("BN2", nn.BatchNorm2d(32)),
                    ("conv3", nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)),
                ]
            )
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass of the ConvNetBN model.

        Args:
            x (torch.tensor): Input tensor. It is expected to have the shape
            `b, 1, h, w`, where `b` is the batch size, `h` is the height, and
            `w` is the width.

        Returns:
            torch.tensor: The output tensor of the ConvNetBN model. It has shape
            `b, 1, h, w`.
        """
        x = self.convnet(x)
        return x


# =============================================================================
class DConvNet(nn.Module):
    """A slightly deeper convolutional neural network model, with batch normalization.

    This model is composed of four convolutional layers. The first three layers
    are followed by a ReLU activation function and a batch normalization layer.
    The last layer does not have any activation function nor batch normalization.

    The kernel sizes of the four layers are 9, 1, 3, and 5. The stride of all
    layers is 1, and the padding of the four layers are 4, 0, 1, and 2, respectively.
    The number of output channels of the first layer is 64, the second layer is 64,
    the third layer is 32, and the fourth layer is 1.

    This class has no arguments.

    Attributes:
        convnet (torch.nn.Sequential): The convolutional neural network model.
        It contains an ordered dictionary with the following keys:
        - 'conv1': The first convolutional layer.
        - 'relu1': The ReLU activation function after the first layer.
        - 'BN1': The batch normalization layer after the first layer.
        - 'conv2': The second convolutional layer.
        - 'relu2': The ReLU activation function after the second layer.
        - 'BN2': The batch normalization layer after the second layer.
        - 'conv3': The third convolutional layer.
        - 'relu3': The ReLU activation function after the third layer.
        - 'BN3': The batch normalization layer after the third layer.
        - 'conv4': The fourth convolutional layer.
    """

    def __init__(self):
        super(DConvNet, self).__init__()
        self.convnet = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4)),
                    ("relu1", nn.ReLU()),
                    ("BN1", nn.BatchNorm2d(64)),
                    ("conv2", nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)),
                    ("relu2", nn.ReLU()),
                    ("BN2", nn.BatchNorm2d(64)),
                    ("conv3", nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)),
                    ("relu3", nn.ReLU()),
                    ("BN3", nn.BatchNorm2d(32)),
                    ("conv4", nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)),
                ]
            )
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass of the DConvNet model.

        Args:
            x (torch.tensor): Input tensor. It is expected to have the shape
            `b, 1, h, w`, where `b` is the batch size, `h` is the height, and
            `w` is the width.

        Returns:
            torch.tensor: The output tensor of the DConvNet model. It has shape
            `b, 1, h, w`.
        """
        x = self.convnet(x)
        return x


# =============================================================================
class List_denoi(nn.Module):
    """A list of layers, which can be denoising layers or any other layers.

    The list consists of `n_denoi` layers of the same type. The forward pass
    of this model takes an input tensor and an iteration number. The iteration
    number is used to select the layer to apply to the input tensor. If the
    iteration number is greater than or equal to `n_denoi`, the last layer is
    applied to the input tensor.

    Each layer can then be modified by the user to have different weights, by
    accessing the layers in the `conv` attribute.

    Attributes:
        n_denoi (int): The number of layers in the list.

        conv (torch.nn.ModuleList): The list of layers.

    Args:
        Denoi (torch.nn.Module): The layer to be repeated in the list.

        n_denoi (int): The number of layers in the list.
    """

    def __init__(self, Denoi, n_denoi):
        super(List_denoi, self).__init__()
        self.n_denoi = n_denoi
        conv_list = []
        for i in range(n_denoi):
            conv_list.append(copy.deepcopy(Denoi))
        self.conv = nn.ModuleList(conv_list)

    def forward(self, x: torch.tensor, iterate: int) -> torch.tensor:
        """Forward pass of the List_denoi model.

        It takes as input an input tensor and an iteration number. The iteration
        number is used to select the layer to apply to the input tensor. If the
        iteration number is greater than or equal to `n_denoi`, the last layer is
        applied to the input tensor.

        Args:
            x (torch.tensor): Input tensor. Its shape is defined by the input
            layer of the `Denoi` layer.

            iterate (int): The iteration number. It is used to select the layer
            to apply to the input tensor.

        Returns:
            torch.tensor: The output tensor of the selected layer.
        """
        index = min(iterate, self.n_denoi - 1)
        x = self.conv[index](x)
        return x


# =============================================================================
class Identity(nn.Module):
    """Identity layer. Can be useful for ablation study.

    This layer returns the input tensor as is.

    This class has no arguments and no attributes.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Forward pass of the Identity layer.

        Args:
            x (Any): Any object (usually torch.tensor).

        Returns:
            Any: The input object.
        """
        return x
