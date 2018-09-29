import torch
import torch.nn as nn


class Residual_Blocks(nn.Module):
    def __init__(self, num_blocks, num_conv_layers, kernel_size, mask = None,
                   num_filters = 128, num_heads = 8,
                   seq_len = None, bias = True, dropout = 0.2):
        super(Residual_Blocks, self).__init__()

        self.residual_blocks = torch.nn.Sequential()
        for i in range(num_blocks):
            self.residual_blocks.a



    def forward(self, input):
        pass



def residual_block(inputs, num_blocks, num_conv_layers, kernel_size, mask = None,
                   num_filters = 128, num_heads = 8,
                   seq_len = None, bias = True, dropout = 0.0):

    outputs = inputs
    sublayer = 1
    total_sublayers = (num_conv_layers + 2) * num_blocks
    for i in range(num_blocks):
        outputs = add_timing_signal_1d(outputs)
        outputs, sublayer = conv_block(outputs, num_conv_layers, kernel_size, num_filters,
                                       seq_len=seq_len, scope="encoder_block_%d" % i, reuse=reuse, bias=bias,
                                       dropout=dropout, sublayers=(sublayer, total_sublayers))
        outputs, sublayer = self_attention_block(outputs, num_filters, seq_len, mask=mask, num_heads=num_heads,
                                                 scope="self_attention_layers%d" % i, reuse=reuse,
                                                 is_training=is_training,
                                                 bias=bias, dropout=dropout, sublayers=(sublayer, total_sublayers))
    return outputs





def conv(inputs, output_size, bias = None, activation = None, kernel_size = 1, name = "conv", reuse = None):
    with tf.variable_scope(name, reuse = reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1,kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,1,output_size]
            strides = [1,1,1,1]
        else:
            filter_shape = [kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype = tf.float32,
                        regularizer=regularizer,
                        initializer = initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs