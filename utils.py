import gensim
import collections
import numpy as np

seq_file_name = 'sequences.utf8'
model_file_name = 'word_vec.utf8'
label2index_dict = {'S': 0, 'B': 1, 'I': 2, 'E':3}
index2label_dict = {0: 'S', 1: 'B', 2: 'I', 3: 'E'}
max_sen_len = 200

'''
line format: '今天晴 BES'
word_seqs format: ['今天晴', '今天阴']
label_seqs format: ['BES', 'BES']
'''
def read_seqs():
    seq_file = open(seq_file_name, 'r', encoding='utf8')
    word_seqs = []
    label_seqs = []
    for line in seq_file:
        line.strip('\n')
        items = line.split(' ')
        if len(items) == 2:
            word_seqs.append(items[0])
            label_seqs.append(items[1].strip('\n'))
    return word_seqs, label_seqs

'''
load word2vec model
generate a word_vec contain all word vectors in model and a blank row with 0 on first row
generate a word2index_dict to construct a function from word to index, if word not found, index is 0
return word_vec and word2index_dict 
'''
def get_word_data():
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file_name, binary=False)
    # one more row at thr first, this row is represent unknown word, and all value in it are 0
    word_vec = np.zeros([len(model.index2word)+1, model.vector_size])
    word2index_dict = dict(UN=0)
    word_index = 1
    for word in model.index2word:
        word_vec[word_index, :] = model.wv[word]
        word2index_dict[word] = word_index
        word_index += 1
    return word_vec, word2index_dict
        
def build_trans_prob(label_seqs):
    label_trans_counter = collections.defaultdict(lambda: 0)
    for label_seq in label_seqs:
        for i in range(1, len(label_seq)):
            label_trans_counter[(label_seq[i-1], label_seq[i])] += 1
    label_trans_prob = np.zeros([4, 4])
    for pre_label, pre_index in label2index_dict.items():
        for label, index in label2index_dict.items():
            label_trans_prob[pre_index, index] = label_trans_counter[(pre_label, label)]
    label_trans_prob = np.divide(label_trans_prob, np.sum(label_trans_prob))
    return label_trans_prob

'''
build data set for train
train_data format: 
[
    {'word_indices': [1,1,1], 'label_indices': [1,1,1]},
    {'word_indices': [1,1,1], 'label_indices': [1,1,1]}
]
'''
def build_train_data(word2index_dict, word_seqs, label_seqs):
    train_data = []
    for word_seq, label_seq in zip(word_seqs, label_seqs):
        word_indices, label_indices = [], []
        if (len(word_seq) == len(label_seq)):
            for word, label in zip(word_seq, label_seq):
                word_index = 0 # default index value 0, unknown word
                if word in word2index_dict:
                    word_index = word2index_dict[word]
                label_index = label2index_dict[label]
                word_indices.append(word_index)
                label_indices.append(label_index)
            train_data.append({'word_indices': word_indices, 'label_indices': label_indices})
    return train_data

def build_test_data(word2index_dict, sentence):
    word_indices = []
    for word in sentence:
        word_index = 0
        if word in word2index_dict:
            word_index = word2index_dict[word]
        word_indices.append(word_index)
    return [word_indices]

'''
get batch with batch size
return data size <= batch_size
give up sentence who's length more than max_sen_len
padding the blank area with 0 in order to make all inputs with same length
'''
def get_batch(train_data, batch_size=200):
    randlist = np.random.randint(low=0, high=len(train_data), size=batch_size)
    words_input = []
    labels_input = []
    for num in randlist:
        sentence = train_data[num]
        sen_len = len(sentence['word_indices'])
        if sen_len <= 200:
            words_input.append(sentence['word_indices'] + [0]*(max_sen_len-sen_len))
            labels_input.append(sentence['label_indices'] + [0]*(max_sen_len-sen_len))
    sen_lens = [max_sen_len] * len(words_input)
    return words_input, labels_input, sen_lens