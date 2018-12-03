import os, re, sys
import gensim, logging
import multiprocessing
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences(object):
    def __init__(self, fname):
        self.fname = fname
        
    def __iter__(self):
        for line in open(self.fname):
            s = re.sub(r'\s{1,2}','',line)
            yield list(s)

#assert len(sys.argv) == 5
sentences = MySentences("sentence.utf8")
model = gensim.models.Word2Vec(sentences, size=300, min_count=5, workers=multiprocessing.cpu_count(), iter=3)
model.wv.save_word2vec_format("vec.utf8",binary=False)