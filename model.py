import numpy as np
import tensorflow as tf

np.random.seed(seed=1)
class Model:
    def __init__(self, sess, word_vec, tran_pro, hidden_size):
        self.train_num = 0
        self.optimizer = tf.train.AdamOptimizer()
        self.sess = sess
        self.words = tf.placeholder(tf.int32, shape=[None, None])
        self.labels = tf.placeholder(tf.int32, shape=[None, None])
        self.sen_lens = tf.placeholder(tf.int32, shape=[None,])
        self.embeds = tf.Variable(word_vec, dtype=tf.float32, trainable=False)
        self.tran_pro = tf.Variable(tran_pro, dtype=tf.float32, trainable=False)       
        self.word_embeds = tf.nn.embedding_lookup(self.embeds, self.words)
        
        fw_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(hidden_size)], state_is_tuple=True)
        bw_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(hidden_size)], state_is_tuple=True)
        
        (fw_out, bw_out), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell, self.word_embeds, sequence_length=self.sen_lens, dtype=tf.float32)
        
        self.weights = tf.get_variable("weights", shape=[hidden_size*2, 4], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.biases = tf.get_variable("biases", shape=[4], dtype=tf.float32, initializer=tf.zeros_initializer())
        
        bi_out = tf.concat([fw_out, bw_out], axis=-1)
        steps = tf.shape(bi_out)[1]
        bi_out_reshape = tf.reshape(bi_out, [-1, hidden_size*2])
        pred = tf.matmul(bi_out_reshape, self.weights) + self.biases
        self.scores = tf.reshape(pred, [-1, steps, 4])
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.scores, self.labels, self.sen_lens, self.tran_pro)
        self.loss = tf.reduce_mean(-log_likelihood)
        
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm( tf.gradients(self.loss, tvars), 5 )
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        
        self.merged = tf.summary.merge_all()

    def step(self, i, data):
        sen_x, sen_y, sen_w, sen_lens = data
        loss, summary, _ = self.sess.run( [ self.loss, self.merged, self.train_op ], feed_dict={self.words: sen_x,
                                                                                                self.labels: sen_y,
                                                                                                self.sen_lens: sen_lens} )
        return loss
  
    def valid(self, data):
        sen_x, sen_y, sen_w, sen_lens = data
        loss = self.sess.run( self.loss, feed_dict={self.words: sen_x,
                                                    self.labels: sen_y,
                                                    self.sen_lens: sen_lens} )
        return loss
  
    def test(self, sentence, label_transition_proba):
        scores = self.sess.run(self.scores,feed_dict={self.words: sentence,
                                                      self.sen_lens: [len(sentence[0])]} )
        viterbi, _ = tf.contrib.crf.viterbi_decode(np.squeeze( scores, 0 ), label_transition_proba)
        return viterbi
  
    def train(self, modelsavepath, data, batch_size=256):
        valid_ids = np.random.randint( low=0, high=len( data ),  size=int(np.ceil(len( data ) / 5)) )
        train_data, valid_data = [], []
        for i in range( len(data) ):
            if i in valid_ids:
                valid_data.append( data[i] )
            else:
                train_data.append( data[i] )
        last_loss=np.inf
        historiesloss = []
        i = 0
        while True:
            loss = self.step(i, self.get_batch(train_data, batch_size))
        
            if i % 3==0:
                self.train_num += 1
                print("trian num: " + str(self.train_num))
                loss = np.mean( [ self.valid( self.get_batch(valid_data[s:s+batch_size]) ) for s in range(0, len(valid_data), batch_size) ]  )
                historiesloss.append(loss)
                if len(historiesloss) >= 4:
                    firstscore = np.mean( historiesloss[-4:-2] )
                    secondscore = np.mean( historiesloss[-2:] )
                    if loss < last_loss:
                        last_loss = loss
                    if firstscore > secondscore:
                        print( "train complate" )
                        break
            i = i + 1
          
    def get_batch(self, data, size=np.inf):
        # get a random batch of data by specified batch_size
        sentences = []
        max_sentence_length = 0
        sentence_lengths = []
        size = min(size, len(data))
        for i in np.random.randint(low=0, high=len(data), size=size):
            sentences.append( data[i] )
            sentence_lengths.append(len(data[i][0]))
        max_sentence_length = np.max(sentence_lengths)
        sentences_x, sentences_y, sentences_w = [], [], []
        for words, labels in sentences:
            sentences_x.append( words + [0] * (max_sentence_length-len(words)) )
            sentences_y.append( labels + [0] * (max_sentence_length-len(labels)) )
            sentences_w.append( [1]*len(labels) + [0]*(max_sentence_length-len(labels)) )
        return sentences_x, sentences_y, sentences_w, sentence_lengths