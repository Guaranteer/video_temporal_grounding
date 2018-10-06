# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
An encoder that conv over embeddings, as described in
https://arxiv.org/abs/1705.03122.
"""


import tensorflow as tf

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def linear_mapping(inputs, out_dim, dropout=1.0, var_scope_name="linear_mapping"):
    with tf.variable_scope(var_scope_name):
        input_shape = inputs.get_shape().as_list()  # static shape. may has None
        return tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=out_dim, activation_fn=None,
                                                 weights_initializer=tf.random_normal_initializer(mean=0,
                                                                                                  stddev=tf.sqrt(
                                                                                                      dropout * 1.0 /
                                                                                                      input_shape[-1])),
                                                 biases_initializer=tf.zeros_initializer())


def linear_mapping_weightnorm(inputs, out_dim, dropout=1.0, var_scope_name="linear_mapping"):
    with tf.variable_scope(var_scope_name):
        input_shape = inputs.get_shape().as_list()  # static shape. may has None
        input_shape_tensor = tf.shape(inputs)
        # use weight normalization (Salimans & Kingma, 2016)  w = g* v/2-norm(v)
        V = tf.get_variable('V', shape=[int(input_shape[-1]), out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                                dropout * 1.0 / int(input_shape[-1]))), trainable=True)
        V_norm = tf.norm(V.initialized_value(), axis=0)  # V shape is M*N,  V_norm shape is N
        g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(),
                            trainable=True)  # weightnorm bias is init zero

        assert len(input_shape) == 3
        inputs = tf.reshape(inputs, [-1, input_shape[-1]])
        inputs = tf.matmul(inputs, V)
        inputs = tf.reshape(inputs, [input_shape_tensor[0], -1, out_dim])
        # inputs = tf.matmul(inputs, V)    # x*v

        scaler = tf.div(g, tf.norm(V, axis=0))  # g/2-norm(v)
        inputs = tf.reshape(scaler, [1, out_dim]) * inputs + tf.reshape(b, [1, out_dim])  # x*v g/2-norm(v) + b

        return inputs


def conv1d_with_bias(inputs, layer_idx, out_dim, kernel_size, padding="SAME", dropout=1.0,
                      var_scope_name="conv_layer"):  # padding should take attention

    with tf.variable_scope("conv_layer_" + str(layer_idx)):
        in_dim = int(inputs.get_shape()[-1])
        V = tf.get_variable('V', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                                4.0 * dropout / (kernel_size * in_dim))), trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
        inputs = tf.nn.bias_add(tf.nn.conv1d(value=inputs, filters=V, stride=1, padding=padding), b)
        return inputs


def gated_linear_units(inputs):
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) == 3
    input_pass = inputs[:, :, 0:int(input_shape[2] / 2)]
    input_gate = inputs[:, :, int(input_shape[2] / 2):]
    input_gate = tf.sigmoid(input_gate)
    return tf.multiply(input_pass, input_gate)


def conv_encoder_stack(inputs, nhids_list, kwidths_list, dropout_dict, mode):
    next_layer = inputs
    for layer_idx in range(len(nhids_list)):
        nin = nhids_list[layer_idx] if layer_idx == 0 else nhids_list[layer_idx - 1]
        nout = nhids_list[layer_idx]
        if nin != nout:
            # mapping for res add
            res_inputs = linear_mapping(next_layer, nout, dropout=dropout_dict['src'],
                                                   var_scope_name="linear_mapping_cnn_" + str(layer_idx))
        else:
            res_inputs = next_layer
        # dropout before input to conv
        next_layer = tf.contrib.layers.dropout(
            inputs=next_layer,
            keep_prob=dropout_dict['hid'],
            is_training=mode)

        next_layer = conv1d_with_bias(inputs=next_layer, layer_idx=layer_idx, out_dim=nout * 2,
                                       kernel_size=kwidths_list[layer_idx], padding="SAME", dropout=dropout_dict['hid'],
                                       var_scope_name="conv_layer_" + str(layer_idx))
        ''' 
        next_layer = tf.contrib.layers.conv2d(
            inputs=next_layer,
            num_outputs=nout*2,
            kernel_size=kwidths_list[layer_idx],
            padding="SAME",   #should take attention
            weights_initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(4 * dropout_dict['hid'] / (kwidths_list[layer_idx] * next_layer.get_shape().as_list()[-1]))),
            biases_initializer=tf.zeros_initializer(),
            activation_fn=None,
            scope="conv_layer_"+str(layer_idx))
        '''
        # next_layer = tf.layers.batch_normalization(next_layer, training=mode)

        next_layer = gated_linear_units(next_layer)
        next_layer = (next_layer + res_inputs) * tf.sqrt(0.5)
        next_layer = normalize(next_layer)




    return next_layer


def conv_decoder_stack(target_embed, enc_output, inputs, nhids_list, kwidths_list, dropout_dict, mode):
    next_layer = inputs
    for layer_idx in range(len(nhids_list)):
        nin = nhids_list[layer_idx] if layer_idx == 0 else nhids_list[layer_idx - 1]
        nout = nhids_list[layer_idx]
        if nin != nout:
            # mapping for res add
            res_inputs = linear_mapping(next_layer, nout, dropout=dropout_dict['hid'],
                                                   var_scope_name="linear_mapping_cnn_" + str(layer_idx))
        else:
            res_inputs = next_layer
        # dropout before input to conv
        next_layer = tf.contrib.layers.dropout(
            inputs=next_layer,
            keep_prob=dropout_dict['hid'],
            is_training=mode)
        # special process here, first padd then conv, because tf does not suport padding other than SAME and VALID
        next_layer = tf.pad(next_layer, [[0, 0], [kwidths_list[layer_idx] - 1, kwidths_list[layer_idx] - 1], [0, 0]],
                            "CONSTANT")

        next_layer = conv1d_with_bias(inputs=next_layer, layer_idx=layer_idx, out_dim=nout * 2,
                                       kernel_size=kwidths_list[layer_idx], padding="VALID",
                                       dropout=dropout_dict['hid'], var_scope_name="conv_layer_" + str(layer_idx))
        '''
        next_layer = tf.contrib.layers.conv2d(
            inputs=next_layer,
            num_outputs=nout*2,
            kernel_size=kwidths_list[layer_idx],
            padding="VALID",   #should take attention, not SAME but VALID
            weights_initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(4 * dropout_dict['hid'] / (kwidths_list[layer_idx] * next_layer.get_shape().as_list()[-1]))),
            biases_initializer=tf.zeros_initializer(),
            activation_fn=None,
            scope="conv_layer_"+str(layer_idx))
        '''
        layer_shape = next_layer.get_shape().as_list()
        assert len(layer_shape) == 3
        # to avoid using future information
        next_layer = next_layer[:, 0:-kwidths_list[layer_idx] + 1, :]

        next_layer = normalize(next_layer)

        next_layer = gated_linear_units(next_layer)

        # add attention
        att_out = make_attention(target_embed, enc_output, next_layer, layer_idx)
        next_layer = (next_layer + att_out) * tf.sqrt(0.5)

        # add res connections
        next_layer += (next_layer + res_inputs) * tf.sqrt(0.5)
    return next_layer


def make_attention(target_embed, encoder_output, decoder_hidden, layer_idx):
    with tf.variable_scope("attention_layer_" + str(layer_idx)):
        embed_size = target_embed.get_shape().as_list()[-1]  # k
        dec_hidden_proj = linear_mapping(decoder_hidden, embed_size,
                                                    var_scope_name="linear_mapping_att_frame")  # M*N1*k1 --> M*N1*k
        ques_hidden_proj = tf.expand_dims(encoder_output.ques_final_state, 1)
        dec_rep = (dec_hidden_proj + target_embed + ques_hidden_proj) * tf.sqrt(0.3333)

        encoder_output_a = encoder_output.frame_final_output
        encoder_output_c = encoder_output.frame_value_output  # M*N2*K


        att_score = tf.matmul(dec_rep, encoder_output_a, transpose_b=True)  # M*N1*K  ** M*N2*K  --> M*N1*N2
        att_score = tf.nn.softmax(att_score)

        length = tf.cast(tf.shape(encoder_output_c), tf.float32)
        att_out = tf.matmul(att_score, encoder_output_c) * length[1] * tf.sqrt(
            1.0 / length[1])  # M*N1*N2  ** M*N2*K   --> M*N1*k

        att_out = linear_mapping(att_out, decoder_hidden.get_shape().as_list()[-1],
                                            var_scope_name="linear_mapping_att_out")
    return att_out


