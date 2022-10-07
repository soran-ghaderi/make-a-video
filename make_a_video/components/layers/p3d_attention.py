import einops
import tensorflow as tf


def flatten(input_tensor):
    return einops.rearrange(input_tensor, 'b c f h w -> b c f (h w)')


def unflatten(input_tensor, h, w):
    return einops.rearrange(input_tensor, '... (h w) -> ... h w', h=h, w=w)


class Attention2D:
    pass


class Attention1D:
    pass


if __name__ == '__main__':
    input = tf.random.normal((10, 3, 7, 5, 5))
    print(input.shape)
    flatten_t = flatten((input))
    print(flatten_t.shape)
    print(unflatten(flatten_t, 5, 5).shape)
