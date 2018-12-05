import tensorflow as tf 
import utils
import numpy as np

class Model:
    def __init__(self, sess, wordvecs, trans_matrix):
        self.sess = sess
        self.hidden_size = 50
        # self.trian_upper = 1
        # self.optimizer = tf.train.AdamOptimizer()
        self.all_wordvecs = tf.Variable(wordvecs, dtype=tf.float32, trainable=False)
        self.trans_matrix = tf.Variable(trans_matrix, dtype=tf.float32, trainable=False)
        self.words_input = tf.placeholder(tf.int32, shape=[None, None])
        self.labels_input = tf.placeholder(tf.int32, shape=[None, None])
        self.sen_lens = tf.placeholder(tf.int32, shape=[None,])
        self.wordvecs_input = tf.nn.embedding_lookup(self.all_wordvecs, self.words_input)

        fw_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.hidden_size)])
        bw_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.hidden_size)])
        # shape of fw_out, bw_out are [batch_size, sen_len, hidden_size]
        (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, 
                                self.wordvecs_input, sequence_length=self.sen_lens, dtype=tf.float32)
        self.weights = tf.get_variable('weights', shape=[self.hidden_size*2, 4], 
                        dtype=tf.float32, initializer=tf.zeros_initializer())
        self.biases = tf.get_variable('biases', shape=[4], dtype=tf.float32, initializer=tf.zeros_initializer())
        # after concat, shape of bi_out is [batch_size, sen_len, hidden_size*2]
        bi_out = tf.concat([fw_out, bw_out], axis=-1)
        bi_out_senlen = tf.shape(bi_out)[1]
        bi_out_reshape = tf.reshape(bi_out, [-1, self.hidden_size*2])
        # bi_out_reshape shape: [batch_size*sen_len, hidden_size*2], weights shape: [hidden_size*2, 4]
        # bi_out_reshape put all other data in dim 1, dim 2 is bi-lstm output data
        # weights shape 4 is number of tags
        fc_out = tf.matmul(bi_out_reshape, self.weights) + self.biases
        self.scores = tf.reshape(fc_out, [-1, bi_out_senlen, 4])
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(self.scores, 
                            self.labels_input, self.sen_lens, self.trans_matrix)
        self.loss = tf.reduce_mean(-log_likelihood)
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
        self.saver = tf.train.Saver(tf.global_variables())
    
    def save(self, savepath='model'):
        self.saver.save(self.sess, savepath)

    def eval(self, words_input, trans_prob):
        scores = self.sess.run(self.scores, feed_dict={self.words_input: words_input, self.sen_lens: [len(words_input[0])]})
        label_seq, _ = tf.contrib.crf.viterbi_decode(np.squeeze(scores, 0), trans_prob)
        return label_seq
        
    def train(self, train_data, train_upper):
        for train_time in range(train_upper):
            print('train time: ' + str(train_time))
            words_input, labels_input, sen_lens = utils.get_batch(train_data)
            self.train_step.run(feed_dict={self.words_input: words_input, self.labels_input: labels_input, 
                                self.sen_lens: sen_lens}, session=self.sess)
            loss = self.loss.eval(feed_dict={self.words_input: words_input, self.labels_input: labels_input,
                                    self.sen_lens: sen_lens}, session=self.sess)
            print('loss: ' + str(loss))
