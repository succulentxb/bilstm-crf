import tensorflow as tf
import model
import utils

label_dict = {'S': 0, 'B': 1, 'I': 2, 'E': 3}
seqs = utils.get_seqs_of('train.utf8')
word_vec, word_dict, label_trans_pro, data = utils.build_dataset("vec.utf8", seqs, label_dict)
sess = tf.Session()
model = model.Model(sess, word_vec, label_trans_pro, 50)

#ckpt = tf.train.get_checkpoint_state( modelsavepath ) 

#model.restore()
sess.run( tf.global_variables_initializer() )
model.train('modelsavepath', data, batch_size=10)

def viterbi(sent):
  label_dict_reverse = {0: 'S', 1: 'B', 2: 'I', 3: 'E'}
  #word_dict, label_reverse_dictionary_
  end_char=''
  if sent[-1]=='\n':
    end_char = '\n'
    sent = sent[:-1]
  if not sent:
    return sent, [], end_char
  sentence = [[word_dict.get(w, 0) for w in sent ]]
  labels = model.test( sentence, label_trans_pro )
  labels = [ label_dict_reverse[i] for i in labels ]
  return sent, labels, end_char


with open( "test.utf8" , 'rt', encoding='utf8') as f:
  test = f.readlines()
for sent in test:
  seq, labels, end = viterbi(sent)
  segment = []
  for char, tag in zip(seq, labels):
    if tag == 'B':
      segment.append(char)
    elif tag == 'M':
      if segment:
        segment[-1] += char
      else:
        segment =  [char]
    elif tag == 'E':
      if segment:
        segment[-1] += char
      else:
        segment =  [char]
    elif tag == 'S':
      segment.append(char)
    else:
      raise Exception()
  print('  '.join(segment), sep='', end=end)
  #break
