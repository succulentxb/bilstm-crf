
if __name__ == "__main__":
    gold_file = open('gold.utf8', 'r')
    target_file = open('target.utf8', 'w')
    for line in gold_file:
        line = line.strip('\n')
        words = line.split(' ')
        '''if words[-1][-1] == '\n':
            words[-1] = words[-1][:-1]'''
        for word in words:
            if len(word) == 1:
                target_file.write(word + ' S\n')
            elif len(word) >= 2:
                target_file.write(word[0] + ' B\n')
                for i in range(1, len(word)-1):
                    target_file.write(word[i] + ' I\n')
                target_file.write(word[-1] + ' E\n')
        target_file.write('\n')