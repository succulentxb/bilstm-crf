def evaluation():
    e = 0
    N = 0
    c = 0
    gold = open('./gold.utf8', 'r', encoding='utf-8')
    pre = open('predict.utf8', 'r', encoding='utf-8')
    gold_lines = gold.readlines()
    pre_lines = pre.readlines()

    for index, l in enumerate(gold_lines):
        l = l.strip().split()
        count = 0
        gold_sub = []
        for item in l:
            sub = []
            for n in range(len(item)):
                sub.append(count)
                count += 1
            gold_sub.append(sub)

        l2 = pre_lines[index].strip().split()
        count = 0
        pre_sub = []
        for item in l2:
            sub = []
            for n in range(len(item)):
                sub.append(count)
                count += 1
            pre_sub.append(sub)
        for x in pre_sub:
            if x in gold_sub:
                c += 1
            else:
                e += 1

        N += len(gold_sub)

    recall = c / N
    precision = c / (c + e)
    f1 = (2 * recall * precision) / (recall + precision)
    er = e / N
    print(' r ', recall, ' p ', precision, ' f1 ', f1, ' er ', er)
if __name__ == '__main__':
    evaluation()