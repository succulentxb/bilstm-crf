
if __name__ == '__main__':
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
        #print(words)
        