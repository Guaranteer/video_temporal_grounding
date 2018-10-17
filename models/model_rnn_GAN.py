import sys
sys.path.append('..')
import json
from dataloaders.dataloader_rnn import Loader
import tensorflow as tf
import tools.layers as layers
import tools.transformer as transformer



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

            q_feature = tf.concat([ques_states[0][1],ques_states[1][1]], 1)
            self.q_feature = tf.contrib.layers.dropout(q_feature, self.dropout, is_training=self.is_training)


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
            logit_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits= logit_score, labels=self.gt_predict)
            avg_logit_loss = tf.reduce_mean(tf.reduce_sum(logit_loss,1))


            self.G_variables = tf.trainable_variables()
            G_regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in self.G_variables])
            G_reg_loss = self.regularization_beta * G_regularization_cost

            ground_prod = tf.nn.softmax(self.gt_predict)
            ground_v_feature = tf.reduce_sum(tf.multiply(model_outputs, tf.expand_dims(ground_prod, 2)), 1)
            ground_out = self.discriminator(ground_v_feature, self.q_feature)

            generated_prod = tf.nn.softmax(logit_score)
            generated_v_feature = tf.reduce_sum(tf.multiply(model_outputs, tf.expand_dims(generated_prod, 2)), 1)
            generated_out = self.discriminator(generated_v_feature, self.q_feature, reuse_flag=True)

            all_variable = tf.trainable_variables()
            self.D_variables = [vv for vv in all_variable if vv not in self.G_variables]
            D_regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in self.D_variables])
            D_reg_loss = self.regularization_beta * D_regularization_cost

            ground_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(ground_out), logits=ground_out))
            generated_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(generated_out), logits=generated_out))


            regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in all_variable])
            self.reg_loss = self.regularization_beta * regularization_cost
            self.dist_loss = avg_logit_loss


            self.G_pre_loss = avg_logit_loss + G_reg_loss
            self.D_loss = ground_loss + generated_loss + D_reg_loss
            scale = self.max_frames/5
            self.G_loss = avg_logit_loss + G_reg_loss + scale*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                                            labels=tf.ones_like(generated_out), logits=generated_out))
            self.frame_score = tf.nn.sigmoid(logit_score)
            print(self.G_variables)
            print(self.D_variables)

    def discriminator(self, v_feature, q_feature, reuse_flag=False):

        with tf.variable_scope("discriminator" , reuse=reuse_flag):
            v_feature = layers.linear_layer(v_feature, self.hidden_size, scope_name='v_transfor')
            # v_feature = tf.contrib.layers.dropout(v_feature, self.dropout, is_training=self.is_training)

            q_feature = layers.linear_layer(q_feature, self.hidden_size, scope_name='q_transfor')
            # q_feature = tf.contrib.layers.dropout(q_feature, self.dropout, is_training=self.is_training)

            fused_add = v_feature + q_feature
            fused_mul = v_feature * q_feature
            fused_cat = tf.concat([fused_add, fused_mul], axis=1)
            fused_all = tf.concat([fused_add, fused_mul, fused_cat],axis=1)
            # fused_all = tf.contrib.layers.dropout(fused_all, self.dropout, is_training=self.is_training)


            with tf.variable_scope("output"):
                scores = layers.linear_layer(fused_all, 1, scope_name='output')
                scores = tf.squeeze(scores,1)
            return scores



if __name__ == '__main__':

    config_file = '../configs/config_rnn.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)



