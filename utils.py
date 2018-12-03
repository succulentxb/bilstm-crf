import gensim
import collections
import numpy as np

def get_seqs_of(train_file_name):
    seqs, result = [['','']], []
    file = open(train_file_name, 'r')
    for line in file:
        items = line.split(' ')
        if len(items) == 2:
            label = items[1].strip('\n')
            label = label.strip()
            word = items[0].strip()
            seqs[-1][0] += word
            seqs[-1][1] += label
        else:
            seqs.append(['',''])
    for seq in seqs:
        if len(seq[0]) == len(seq[1]):
            result.append((seq[0], seq[1]))
    return result

def build_dataset(vec_file_name, seqs, label_dict):
    model = gensim.models.KeyedVectors.load_word2vec_format(vec_file_name, binary=False)
    word_vec = np.zeros([len(model.index2word)+1, model.vector_size])
    word_dict = dict(UNK=0)
    i = 1
    for word in model.index2word:
        word_vec[len(word_dict), :] = model.wv[word]
        word_dict[word] = i
        i += 1
    
    label_trans_count = collections.defaultdict(lambda: 0)
    for seq in seqs:
        i = 0
        for word, label in zip(*seq):
            if i > 0:
                label_trans_count[(seq[1][i-1],seq[1][i])] += 1
            i += 1

        
    data = []
    for seq in seqs:
        word_row, label_row = [], []
        for word, label in zip(*seq):
            if word in word_dict:
                word_index = word_dict[word]
            else:
                word_index = 0
            label_index = label_dict[label]
            word_row.append(word_index)
            label_row.append(label_index)
        data.append([word_row, label_row])
    
    label_trans_prob = np.zeros([4, 4])
    for pre_label, pre_index in label_dict.items():
        for label, index in label_dict.items():
            label_trans_prob[pre_index, index] = label_trans_count[(pre_label, label)]  
    label_trans_prob = np.divide(label_trans_prob, np.sum(label_trans_prob) )
    
    return word_vec, word_dict, label_trans_prob, data
