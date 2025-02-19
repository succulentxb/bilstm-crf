def tackle_train_data():
    train_file = open('train.utf8', 'r')
    seq_file = open('sequences.utf8', 'w')
    word_seq = ''
    tag_seq = ''
    for line in train_file:
        line = line.strip('\n')
        line = line.strip()
        items = line.split(' ')
        if len(items) == 2:
            word_seq += items[0]
            tag_seq += items[1]
        else:
            seq_file.write(word_seq + ' ' + tag_seq +'\n')
            word_seq = ''
            tag_seq = ''
    train_file.close()
    seq_file.close()
            
def train2sen():
    train_file = open('train.utf8', 'r')
    target_file = open('sentence.utf8', 'w')
    words = []
    for line in train_file:
        line = line.strip('\n')
        if line != '':
            words.append(line[0])
        else:
            for word in words:
                target_file.write(word + "")
            target_file.write('\n')
            words = []
    train_file.close()
    target_file.close()

def to_test_format(result_file_name):
    result_file = open(result_file_name, 'r', encoding='utf8')
    predict_file = open('predict.utf8', 'w', encoding='utf8')
    wordseq = []
    labelseq = []
    for line in result_file:
        line = line.strip('\n')
        if line == '':
            seged_sen = []
            seged_word = ''
            for i in range(len(wordseq)):
                if labelseq[i] == 'S':
                    seged_word = wordseq[i]
                    seged_sen.append(seged_word)
                    seged_word = ''
                elif labelseq[i] == 'B':
                    seged_word += wordseq[i]
                    if i+1 < len(wordseq):
                        if labelseq[i+1] == 'S' or labelseq[i+1] == 'B':
                            seged_sen.append(seged_word)
                            seged_word = ''
                elif labelseq[i] == 'I':
                    seged_word += wordseq[i]
                elif labelseq[i] == 'E':
                    seged_word += wordseq[i]
                    seged_sen.append(seged_word)
                    seged_word = '' 
            predict_file.write(' '.join(seged_sen) + '\n')
            wordseq = []
            labelseq = []
        else:
            items = line.split(' ')
            wordseq.append(items[0])
            labelseq.append(items[1])

if __name__ == "__main__":
    trans_file_name = input('enter result file name:\n')
    to_test_format(trans_file_name)
        