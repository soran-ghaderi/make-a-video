import os
from typing import Optional, Tuple

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

__all__ = ['P3D']


class P3D:
    def __init__(self, output_channels: int = None, stride: Optional[Tuple[int, Tuple]] = 1,
                 padding: Optional[str] = 'valid'):
        """

        Parameters
        ----------
        output_channels :
            Number of channels produced by the convolution
        stride :
            Stride of the convolution. Default: 1
        padding :
            Padding added to all six sides of the input
        """
        self.s_padding = tf.constant([[0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [0, 0]])
        self.t_padding = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0]])
        self.output_channels = output_channels
        self.stride = stride
        self.padding = padding


    def conv_S(self):
        """Applies a 3D convolution over an input signal composed of several input planes.

        A (1, 3, 3) convolution layer as described in [1]_

        References
        ----------
        [1] https://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf

        Returns
        -------
        out:
            A tensor of rank 5+ representing
        """
        return tf.keras.layers.Conv3D(filters=self.output_channels,
                                      kernel_size=[1, 3, 3],
                                      strides=self.stride,
                                      padding=self.padding,
                                      use_bias=False)

    def conv_T(self):
        """Apply a 3D convolution over an input signal composed of several input planes.

        A (1, 3, 3) convolution layer as described in [1]_

        References
        ----------
        [1] https://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf

        Returns
        -------
        out:
            A tensor of rank 5+ representing
        """

        return tf.keras.layers.Conv3D(filters=self.output_channels,
                                      kernel_size=[3, 1, 1],
                                      strides=self.stride,
                                      padding=self.padding,
                                      use_bias=False)

    def p3d_a(self, output_channels, inputs):
        """Return P3D-A as described in [1]_

        References
        ----------
        [1] https://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf

        Parameters
        ----------
        output_channels :
            Number of channels produced by the convolution
        inputs :
            5+d input tensor
        Returns
        -------
            out:
                A tensor of rank 5+ representing
        """
        spacial = self.conv_S()
        temporal = self.conv_T()

        spacial_padded_inputs = tf.pad(inputs, self.s_padding)
        spacial_p3d = spacial(spacial_padded_inputs)
        spacial_p3d = tf.keras.layers.BatchNormalization(output_channels)(spacial_p3d)
        spacial_p3d = tf.nn.relu(spacial_p3d)

        temporal_padded_inputs = tf.pad(spacial_p3d, self.t_padding)
        temporal_p3d = temporal(temporal_padded_inputs)
        temporal_p3d = tf.keras.layers.BatchNormalization(output_channels)(temporal_p3d)
        temporal_p3d = tf.nn.relu(temporal_p3d)

        return temporal_p3d

    def p3d_b(self, output_channels, inputs):
        """Return P3D-B as described in [1]_

        References
        ----------
        [1] https://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf

        Parameters
        ----------
        output_channels :
            Number of channels produced by the convolution
        inputs :
            5+d input tensor
        Returns
        -------
            out:
                A tensor of rank 5+ representing
        """
        spacial = self.conv_S()
        temporal = self.conv_T()

        spacial_padded_inputs = tf.pad(inputs, self.s_padding)
        spacial_p3d = spacial(spacial_padded_inputs)
        spacial_p3d = tf.keras.layers.BatchNormalization(output_channels)(spacial_p3d)
        spacial_p3d = tf.nn.relu(spacial_p3d)

        temporal_padded_inputs = tf.pad(inputs, self.t_padding)  # raw 'inputs' is fed
        temporal_p3d = temporal(temporal_padded_inputs)
        temporal_p3d = tf.keras.layers.BatchNormalization(output_channels)(temporal_p3d)
        temporal_p3d = tf.nn.relu(temporal_p3d)

        return temporal_p3d + spacial_p3d

    def p3d_c(self, output_channels, inputs):
        """Return P3D-C as described in [1]_

        References
        ----------
        [1] https://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf

        Parameters
        ----------
        output_channels :
            Number of channels produced by the convolution
        inputs :
            5+d input tensor
        Returns
        -------
            out:
                A tensor of rank 5+ representing
        """
        spacial = self.conv_S()
        temporal = self.conv_T()

        spacial_padded_inputs = tf.pad(inputs, self.s_padding)
        spacial_p3d = spacial(spacial_padded_inputs)
        spacial_p3d = tf.keras.layers.BatchNormalization(output_channels)(spacial_p3d)
        spacial_p3d = tf.nn.relu(spacial_p3d)

        temporal_padded_inputs = tf.pad(spacial_p3d, self.t_padding)  # spacial_p3d is fed
        temporal_p3d = temporal(temporal_padded_inputs)
        temporal_p3d = tf.keras.layers.BatchNormalization(output_channels)(temporal_p3d)
        temporal_p3d = tf.nn.relu(temporal_p3d)

        return temporal_p3d + spacial_p3d



if __name__ == '__main__':
    p3d = P3D(5)
    # cs = p3d.conv_S(16)
    input = tf.random.normal((20, 7, 20, 10, 50, 3))

    print("input shape: ", input.shape)
    print("p3d_a shape: ", p3d.p3d_a(5, input).shape)
    print("p3d_b shape: ", p3d.p3d_b(5, input).shape)
    print("p3d_c shape: ", p3d.p3d_c(5, input).shape)
