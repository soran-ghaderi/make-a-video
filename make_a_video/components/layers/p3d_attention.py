import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Union
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"







class P3D:

    def conv_S(self, output_channels: int, stride: Optional[Tuple[int, Tuple]] = 1,
               padding: Optional[str] = 'valid'):
        """Applies a 3D convolution over an input signal composed of several input planes.

        A (1, 3, 3) convolution layer as described in [1]_

        References
        ----------
        [1] https://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf

        Parameters
        ----------
        output_channels :
            Number of channels produced by the convolution
        stride :
            Stride of the convolution. Default: 1
        padding :
            Padding added to all six sides of the input

        Returns
        -------
        out:
            A tensor of rank 5+ representing
        """
        return tf.keras.layers.Conv3D(output_channels, kernel_size=[1, 3, 3], strides=stride, padding=padding,
                                      use_bias=False)

    def conv_T(self, output_channels, stride=1, padding='valid'):
        """Applies a 3D convolution over an input signal composed of several input planes.

        A (1, 3, 3) convolution layer as described in [1]_

        References
        ----------
        [1] https://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf

        Parameters
        ----------
        output_channels :
            Number of channels produced by the convolution
        stride :
            Stride of the convolution. Default: 1
        padding :
            Padding added to all six sides of the input

        Returns
        -------
        out:
            A tensor of rank 5+ representing
        """

        return tf.keras.layers.Conv3D(output_channels, kernel_size=[3, 1, 1], strides=stride, padding=padding,
                                      use_bias=False)

    def p3d_a(self, output_channels, inputs):
        spacial = self.conv_S(output_channels)
        temporal = self.conv_T(output_channels)

        return temporal(spacial(inputs))






if __name__ == '__main__':
    p3d = P3D()
    cs = p3d.conv_S(16)
    input = tf.random.normal((20, 7, 20, 10, 50, 3))
    print(input.shape)
    # print("Shape of the", cs(input).shape)
    print(p3d.p3d_a(5, input).shape)
    ct = p3d.conv_T(16)
    # print(input.shape)
    # print("Shape of the", ct(cs(input)).shape)




