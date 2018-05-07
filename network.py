from typing import Tuple, TypeVar

import tensorflow as tf
from keras import backend as K
from keras.layers import concatenate, Conv2D, Conv2DTranspose, Input, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam

if K.backend() == 'tensorflow':
    K.set_image_data_format('channels_last')
    TensorType = TypeVar(tf.Tensor)
else:
    raise ValueError('This package only works with Tensorflow at the moment.')


class Network:
    def __init__(self, input_width: int, input_height: int, input_channels: int, network_channel_sizes: Tuple[int, ...], channels_last: bool = True,
                 conv_padding: str = 'same', down_conv_kernel: Tuple[int, int] = (3, 3), up_conv_kernel: Tuple[int, int] = (2, 2),
                 up_conv_kernel_strides: Tuple[int, int] = (2, 2), leaky_alpha=0.3, pool_size: Tuple[int, int] = (2, 2), pool_strides: int = 2):
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.inputs = Input(shape=(input_width, input_height, input_channels))
        self.network_channel_sizes = network_channel_sizes
        self.down_conv_kernel = down_conv_kernel
        self.up_conv_kernel = up_conv_kernel
        self.up_conv_kernel_strides = up_conv_kernel_strides
        self.leaky_alpha = leaky_alpha
        self.conv_padding = conv_padding
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.model = None
        if not channels_last:
            K.set_image_data_format('channels_first')

    def _get_downsample_group(self, input_to_group: TensorType, num_filters: int) -> Tuple[TensorType, TensorType]:
        """
        The contracting path follows the typical architecture  of a convolu-
        tional network. It consists of the repeated application of two 3x3 convo-
        lutions (unpadded convolutions), each followed by a rectified linear unit
        (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At
        each downsampling step we double the number of feature channels.
        """

        conv_1 = Conv2D(filters=num_filters, kernel_size=self.down_conv_kernel, padding=self.conv_padding)(input_to_group)
        conv_1 = LeakyReLU(alpha=self.leaky_alpha)(conv_1)
        conv_2 = Conv2D(filters=num_filters, kernel_size=self.down_conv_kernel, padding=self.conv_padding)(conv_1)
        conv_2 = LeakyReLU(alpha=self.leaky_alpha)(conv_2)
        output = MaxPooling2D(pool_size=self.pool_size, strides=self.pool_strides, padding='valid')(conv_2)

        # Output becomes next group's input and conv_2 gets linked to the corresponding upsampling group
        return output, conv_2

    def _get_upsample_group(self, input_to_group: TensorType, num_filters: int, trans_conv_filters: int) -> TensorType:
        """
        Every step in the expansive path consists of an upsampling of the feature
        map followed by a 2x2 convolution (“up-convolution”) that halves the number
        of feature channels, a concatenation with the correspondingly cropped
        feature map from the contracting path, and two 3x3 convolutions, each
        followed by a ReLU
        """

        conv_1 = Conv2D(filters=num_filters, kernel_size=self.down_conv_kernel, padding=self.conv_padding)(input_to_group)
        conv_1 = LeakyReLU(alpha=self.leaky_alpha)(conv_1)
        conv_2 = Conv2D(filters=num_filters, kernel_size=self.down_conv_kernel, padding=self.conv_padding)(conv_1)
        conv_2 = LeakyReLU(alpha=self.leaky_alpha)(conv_2)

        return Conv2DTranspose(filters=trans_conv_filters, kernel_size=self.up_conv_kernel, strides=self.up_conv_kernel_strides)(conv_2)

    def get_model(self):
        # Left side
        output_0, link_0 = self._get_downsample_group(self.inputs, self.network_channel_sizes[0])
        output_1, link_1 = self._get_downsample_group(output_0, self.network_channel_sizes[1])
        output_2, link_2 = self._get_downsample_group(output_1, self.network_channel_sizes[2])
        output_3, link_3 = self._get_downsample_group(output_2, self.network_channel_sizes[3])

        # Right side
        up_1 = concatenate([self._get_upsample_group(output_3, self.network_channel_sizes[-1], self.network_channel_sizes[-2]), link_3], axis=3)
        up_2 = concatenate([self._get_upsample_group(up_1, self.network_channel_sizes[-2], self.network_channel_sizes[-3]), link_2], axis=3)
        up_3 = concatenate([self._get_upsample_group(up_2, self.network_channel_sizes[-3], self.network_channel_sizes[-4]), link_1], axis=3)
        up_4 = concatenate([self._get_upsample_group(up_3, self.network_channel_sizes[-4], self.network_channel_sizes[-5]), link_0], axis=3)

        # Output group
        conv_1 = Conv2D(self.network_channel_sizes[0], kernel_size=self.down_conv_kernel, padding='same')(up_4)
        conv_2 = Conv2D(self.network_channel_sizes[0], kernel_size=self.down_conv_kernel, padding='same')(conv_1)
        outputs = Conv2D(1, (1, 1))(conv_2)
        outputs = LeakyReLU(alpha=self.leaky_alpha)(outputs)

        self.model = Model(inputs=[self.inputs], outputs=[outputs])
        self.model.compile(optimizer=Adam(lr=1e-5), loss='mean_squared_error', metrics=['mse'])

        return self.model
