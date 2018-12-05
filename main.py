import tensorflow as tf
import model
import utils

if __name__ == "__main__":
    print('reading sequences...')
    word_seqs, label_seqs = utils.read_seqs()
    print('loading word2vec model...')
    word_vec, word2index_dict = utils.get_word_data()
    print('buiding trans probality matrix...')
    label_trans_prob = utils.build_trans_prob(label_seqs)
    print('buiding train data...')
    train_data = utils.build_train_data(word2index_dict, word_seqs, label_seqs)

    model_path = 'model'
    sess = tf.Session()
    model = model.Model(sess, word_vec, label_trans_prob)
    sess.run(tf.global_variables_initializer())
    print('start train model...')
    model.train(train_data, 300)
    '''model.save(model_path)
    ckpt = tf.train.get_checkpoint_state(model_path)
    model.saver.restore(sess, ckpt.model_checkpoint_path)
    model.train()'''

    '''test_data = utils.build_test_data(word2index_dict, '今天是个大晴天')
    res = model.eval(test_data, label_trans_prob)
    print([utils.index2label_dict[i] for i in res])'''

    while True:
        test_file_name = input('enter test file name: \n')
        if test_file_name == 'exit':
            exit()
        result_file_name = 'result.utf8'
        result_file = open(result_file_name, 'w')
        test_file = open(test_file_name, 'r')
        test_wordseq = []
        sen_num = 0
        for line in test_file:
            line = line.strip('\n')
            if line == '':
                test_data = utils.build_test_data(word2index_dict, test_wordseq)
                res_labelseq = model.eval(test_data, label_trans_prob)
                res_labelseq = [utils.index2label_dict[i] for i in res_labelseq]
                for word, label in zip(test_wordseq, res_labelseq):
                    result_file.write(word + ' ' + label +'\n')
                result_file.write('\n')
                sen_num += 1
                print('eval ' + str(sen_num) + ' sentences')
                test_wordseq = []
            else:
                test_wordseq.append(line[0])
        test_file.close()
        result_file.close()
                
