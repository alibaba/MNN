#
#  classficationTopkEval.cpp
#  MNN
#
#  Created by MNN on 2019/07/30.
#  Copyright © 2018, Alibaba Group Holding Limited
#
""" Convert ILSVRC devkit validation ground truth label to class ID."""

import sys

def get_id(dic, label):
    cnt = 0
    len_dict = len(dic)
    for i in range(len_dict):
        if dic[i] == label:
            return cnt
        else:
            cnt += 1

    print("Can't find label: ", label)
    assert False

def main(synset_words_file, lables_file):
    synset_words = open(synset_words_file, 'r')
    synset_words_dict = []

    cnt = 0
    for line in synset_words.readlines():
        l = line.strip('\n')
        synset_words_dict.append(l)
        cnt += 1
    synset_words.close()
    lable_id = open('class_id.txt', 'w')

    lables = open(lables_file, 'r')
    for line in lables.readlines():
        l = line.strip('\n')
        id = get_id(synset_words_dict, l)
        lable_id.write(str(id) + '\n')

    lables.close()
    lable_id.close()
    return 0




if __name__ == '__main__':
    synset_words_file = sys.argv[1]
    labels_file = sys.argv[2]
    main(synset_words_file,labels_file)