import gensim, logging
import multiprocessing

class Sentences(object):
    def __init__(self, fname):
        self.fname = fname
        
    def __iter__(self):
        for line in open(self.fname):
            line.strip('\n')
            line.strip(' ')
            yield list(line)

if __name__ == "__main__":
    sentences = Sentences("sentence.utf8")
    model = gensim.models.Word2Vec(sentences, size=300, min_count=0, workers=multiprocessing.cpu_count())
    model.wv.save_word2vec_format("word_vec.utf8",binary=False)
    print('done!')