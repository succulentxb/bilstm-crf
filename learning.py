# -*- coding: utf-8 -*-
import sys, time
import numpy as np
import tensorflow as tf
from model import *
import gensim
import collections

# assert len(sys.argv) == 6

def getTextSequences(filename):
  sequences, result = [['','']], []
  with open(filename, 'rt', encoding='utf8') as f:
    for line in f.readlines():
      if len(line.split('\t')) == 2:
        word, label =  line.split('\t')
        label = label.strip()
        sequences[-1][0] += word
        sequences[-1][1] += label
      else:
        sequences.append(['',''])
  for sequence in sequences:
    if len(sequence[0]) == len(sequence[1]) > 0:
      result.append( (sequence[0], sequence[1]) )
  return result

def build_dataset(word2vecfname, sequences):
  model = gensim.models.KeyedVectors.load_word2vec_format(word2vecfname, binary=False)
  char_vector = np.zeros([len(model.index2word)+1, model.vector_size])
  char_dictionary = dict(UNK=0)
  for w in model.index2word:
    char_vector[len(char_dictionary), :] = model.wv[w]
    char_dictionary[w] = len(char_dictionary)
    
  label_translation_count = collections.defaultdict(lambda: 0)
  label_dictionary = dict()  
  for sequence in sequences:
    for i, (character, label) in enumerate(zip(*sequence)):
      if i>0: label_translation_count[(sequence[1][i-1],sequence[1][i])] += 1
      if label not in label_dictionary:
        label_dictionary[label] = len(label_dictionary)
        
  data = []
  for sequence in sequences:
    charrow, labelrow = [], []
    for character, label in zip(*sequence):
      if character in char_dictionary:
        char_index = char_dictionary[character]
      else:
        char_index = 0 # dictionary['UNK']
      label_index = label_dictionary[label]
      charrow.append( char_index )
      labelrow.append( label_index )
    data.append( [charrow, labelrow] )
    
  label_transition_proba = np.zeros([len(label_dictionary), len(label_dictionary)])
  for prev_label, prev_index in label_dictionary.items():
    for label, index in label_dictionary.items():
      label_transition_proba[prev_index, index] = label_translation_count[(prev_label, label)]  
  label_transition_proba = np.divide(label_transition_proba, np.sum(label_transition_proba) )
    
  char_reverse_dictionary = dict(zip(char_dictionary.values(), char_dictionary.keys()))
  label_reverse_dictionary = dict(zip(label_dictionary.values(), label_dictionary.keys()))
  return char_vector, char_dictionary, char_reverse_dictionary,label_dictionary,label_reverse_dictionary,label_transition_proba,data


modelsavepath = "model"
filename = "train.utf8"
hidden_size=50
batch_size=10
word2vec="vec.utf8"

_starTime = time.time()
print( 'load data ... ', end='', flush=True)
sequences_ = getTextSequences(filename)
print( 'sequences total:%d time:%fs' % (len(sequences_),time.time() - _starTime), flush=True)

_starTime = time.time()
print( 'build dataset ... ', end='', flush=True)
char2vector_, char_dictionary_, char_reverse_dictionary_, label_dictionary_, label_reverse_dictionary_, label_transition_proba_, data_ = build_dataset(word2vec, sequences_)
print( 'chartable size:%d time:%fs' % (len(char_dictionary_),time.time() - _starTime), flush=True)

sess = tf.Session()

model = Model(sess=sess, char_vec=char2vector_, trans_mat=label_transition_proba_, hidden_size=hidden_size)

print(' train and save model ')
sess.run( tf.global_variables_initializer() )
model.train(modelsavepath, data_, batch_size=batch_size)

print( 'execute time: %fs' % (time.time() - _starTime) )
