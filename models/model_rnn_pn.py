import sys
sys.path.append('..')
import json
from dataloaders.dataloader_rnn import Loader
import tensorflow as tf
import tools.layers as layers
import tools.transformer as transformer
import tools.conv_utils as conv_utils



class Model(object):

    def __init__(self, params):

        self.params = params
        self.regularization_beta = params['regularization_beta']
        self.max_frames = params['max_frames']
        self.input_video_dim = params['input_video_dim']
        self.max_words = params['max_words']
        self.input_ques_dim = params['input_ques_dim']
        self.hidden_size = params['hidden_size']
        self.dropout = params['dropout_prob']
        self.regularization_beta = params['regularization_beta']


        self.build_model()


    def build_model(self):
        # input layer (batch_size, n_steps, input_dim)
        self.ques_vecs = tf.placeholder(tf.float32, [None, self.max_words, self.input_ques_dim])
        self.ques_len = tf.placeholder(tf.int32, [None])
        self.frame_vecs = tf.placeholder(tf.float32, [None, self.max_frames, self.input_video_dim])
        self.frame_len = tf.placeholder(tf.int32, [None])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.is_training = tf.placeholder(tf.bool)
        self.gt_predict = tf.placeholder(tf.float32, [None, self.max_frames])
        self.gt_windows = tf.placeholder(tf.float32, [None, 2])

        self.frame_mask = tf.sequence_mask(self.frame_len, maxlen=self.max_frames)
        self.ques_mask = tf.sequence_mask(self.ques_len, maxlen=self.max_words)



        with tf.variable_scope("Frame_Embedding_Encoder_Layer"):
            input_frame_vecs = tf.contrib.layers.dropout(self.frame_vecs, self.dropout, is_training=self.is_training)
            frame_embedding, _ = layers.dynamic_origin_bilstm_layer(input_frame_vecs, self.hidden_size, 'frame_embedding',
                                                                input_len=self.frame_len)


            frame_embedding = tf.contrib.layers.dropout(frame_embedding, self.dropout, is_training=self.is_training)

        with tf.variable_scope("Ques_Embedding_Encoder_Layer"):

            input_ques_vecs = tf.contrib.layers.dropout(self.ques_vecs, self.dropout, is_training=self.is_training)
            ques_embedding, ques_states = layers.dynamic_origin_bilstm_layer(input_ques_vecs, self.hidden_size, 'ques_embedding',
                                                                           input_len=self.ques_len)

            ques_embedding = tf.contrib.layers.dropout(ques_embedding, self.dropout, is_training=self.is_training)




        with tf.variable_scope("Context_to_Query_Attention_Layer"):

            # att_score = tf.matmul(frame_embedding, ques_embedding, transpose_b=True)  # M*N1*K  ** M*N2*K  --> M*N1*N2
            # att_score = tf.nn.softmax(mask_logits(att_score, mask=tf.expand_dims(self.ques_mask, 1)))
            #
            # length = tf.cast(tf.shape(ques_embedding), tf.float32)
            # att_out = tf.matmul(att_score, ques_embedding) * length[1] * tf.sqrt(
            #     1.0 / length[1])  # M*N1*N2  ** M*N2*K   --> M*N1*k
            #
            # attention_outputs = tf.concat([frame_embedding, att_out, tf.multiply(frame_embedding,att_out)])

            att_score = tf.matmul(frame_embedding, ques_embedding, transpose_b=True)  # M*N1*K  ** M*N2*K  --> M*N1*N2
            mask_q = tf.expand_dims(self.ques_mask, 1)
            S_ = tf.nn.softmax(layers.mask_logits(att_score, mask = mask_q))
            mask_v = tf.expand_dims(self.frame_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(layers.mask_logits(att_score, mask = mask_v), axis = 1),(0,2,1))
            self.v2q = tf.matmul(S_, ques_embedding)
            self.q2v = tf.matmul(tf.matmul(S_, S_T), frame_embedding)
            attention_outputs = tf.concat([frame_embedding, self.v2q, frame_embedding * self.v2q, frame_embedding * self.q2v], 2)

        with tf.variable_scope("Model_Encoder_Layer"):
            attention_outputs = tf.contrib.layers.dropout(attention_outputs, self.dropout, is_training=self.is_training)
            model_outputs, _ = layers.dynamic_origin_bilstm_layer(attention_outputs, self.hidden_size, 'model_layer',
                                                                    input_len=self.frame_len)

            model_outputs = tf.contrib.layers.dropout(model_outputs, self.dropout, is_training=self.is_training)


        with tf.variable_scope("Output_Layer"):

            logit_score = layers.linear_layer_3d(model_outputs, 1, scope_name='output_layer')
            logit_score = tf.squeeze(logit_score, 2)
            logit_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits= logit_score, labels=self.gt_predict/ tf.reduce_sum(self.gt_predict, keepdims=True,axis=1))

            # logit_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits= logit_score, labels=self.gt_predict)
            avg_logit_loss = tf.reduce_mean(tf.reduce_sum(logit_loss,1))

            self.G_variables = tf.trainable_variables()
            G_regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in self.G_variables])
            self.test_loss = avg_logit_loss
            self.loss = avg_logit_loss + self.regularization_beta * G_regularization_cost
            self.frame_score = tf.nn.sigmoid(logit_score)


        with tf.variable_scope('Pointer_Layer'):
            score_dist = tf.nn.softmax(logit_score)
            output = tf.nn.relu(conv_utils.conv1d_with_bias(tf.expand_dims(score_dist,2),1,16,5))
            # output = tf.contrib.layers.dropout(output, self.dropout, is_training=self.is_training)
            output = tf.nn.relu(conv_utils.conv1d_with_bias(output,2,32,10))
            # output = tf.contrib.layers.dropout(output, self.dropout, is_training=self.is_training)
            output = tf.nn.relu(conv_utils.conv1d_with_bias(output, 3, 64, 20))
            # output = tf.contrib.layers.dropout(output, self.dropout, is_training=self.is_training)
            output = tf.nn.relu(conv_utils.conv1d_with_bias(output,4,1,10))
            # output = tf.contrib.layers.dropout(output, self.dropout, is_training=self.is_training)
            # output = tf.nn.relu(layers.linear_layer(output, self.hidden_size,scope_name='pointer_1'))
            # output = tf.nn.relu(layers.linear_layer(output, self.hidden_size/2,scope_name='pointer_2'))
            # output = tf.nn.relu(layers.linear_layer(output, self.hidden_size/4,scope_name='pointer_3'))
            output = layers.linear_layer(tf.squeeze(output,2), 2, scope_name='pointrt_output')
            self.predict_start_end = output
            gt_start_end = self.gt_windows
            pointer_loss = tf.reduce_mean(tf.square(tf.subtract(self.predict_start_end, gt_start_end)))

            all_variable = tf.trainable_variables()
            # self.pn_variables = all_variable
            self.pn_variables = [vv for vv in all_variable if vv not in self.G_variables]
            pn_regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in self.pn_variables])
            self.pn_loss = pointer_loss + self.regularization_beta * pn_regularization_cost


if __name__ == '__main__':

    config_file = '../configs/config_rnn.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)



