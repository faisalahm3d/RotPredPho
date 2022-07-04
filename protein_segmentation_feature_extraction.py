import numpy as np
X_train = []
y_train = []

all_sites = []
all_sites_label = []

WINDOW_SIZE = 10
STRUCTURAL_WINDOW_SIZE = 10

skip_sequence_T_test =[48]

ptm_site ='S'


def read_pssm_file(pssmfile):
    fo = open(pssmfile, "r+")
    str = fo.name + fo.read()
    fo.close()
    str = str.split()
    p = str[0:22]
    lastpos = str.index('Lambda')
    lastpos = lastpos - (lastpos % 62) - 4
    currentpos = str.index('Last') + 62
    p_seq = ''
    plen = 0
    pssm = {}
    while (currentpos < lastpos):
        p_no = [int(i) for i in str[currentpos]]
        p_seq = p_seq + str[currentpos + 1]
        pssm[plen] = [int(i) for i in str[currentpos + 2:currentpos + 22]]
        currentpos = currentpos + 44
        plen = plen + 1
    # end while
    return p_seq, pssm, plen


def find_all_indexes(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


def get_pssm_profile_bigram(PSSM, flatten = False):
    #WINDOW_SIZE = 20
    B = [[0 for x in range(20)] for y in range(20)]
    for p in range(20):
        for q in range(20):
            now = 0
            for k in range(0, WINDOW_SIZE * 2):
                now += PSSM[k][p] * PSSM[k + 1][q]
            B[p][q] = now

    return np.asarray(B).flatten() if flatten else B


def get_structural_feature(spd_file, protein_length, ind, flatten = True):
    with open(spd_file) as fp:
        # Read 3 lines for the top padding lines
        fp.readline()
        SSpres = fp.readlines()
        SSpre_list = []
        protein_sz = protein_length
        #STRUCTURAL_WINDOW_SIZE = 20
    for val in range(ind - STRUCTURAL_WINDOW_SIZE, ind + STRUCTURAL_WINDOW_SIZE + 1):
        now = val
        if val < 0 or val >= protein_sz:
            distance = ind - val
            now = ind + distance
        row = list(map(float, SSpres[now].strip().split()[3:]))
        SSpre_list.append(row)
    return np.asarray(SSpre_list).flatten() if flatten else SSpre_list


def get_feature(protein_length, pssm, spd, positive_index, all_index):
    temp = WINDOW_SIZE+1
    new_psitive = [ x-1 for x in positive_index]
    for idx in all_index:
        segmented_pssm = []
        for i in range (WINDOW_SIZE,-temp,-1):
            if idx-i in pssm.keys():
                segmented_pssm.append(pssm[idx-i])
            else: segmented_pssm.append(pssm[idx+i])
        pssm_profile_bigram = get_pssm_profile_bigram(segmented_pssm,flatten=True)
        spd_feature = get_structural_feature(spd,protein_length,idx)
        feature = np.concatenate((pssm_profile_bigram,spd_feature),axis=0)
        X_train.append(feature)
        if idx in new_psitive:
            y_train.append(1)
        else: y_train.append(0)


def get_indexes(amino_acid):
    count = 0
    seq_number = 1
    while True:
        count += 1
        # Get next line from file
        line = file.readline()
        if count % 2 == 1:
            test = line.split()
            # print(test)
            positive_index = []
            for i in range(1, len(test) - 1):
                # string_index = test[i]
                positive_index.append(int(test[i][1:]))
            # print(positive_index)
        else:
            # if seq_number in skip_sequence_T_test:
            #     seq_number+=1
            #     continue
            pssm_file_path = 'dataset/pssm-{}/pssm{}.txt'.format(ptm_site,str(seq_number))
            spd_file_path = 'dataset/spd-{}/pssm{}.spd3'.format(ptm_site,str(seq_number))
            p_seq, pssm_dic, plen_t = read_pssm_file(pssm_file_path)
            indexes = find_all_indexes(p_seq, amino_acid)
            #segment_pssm(pssm_dic,positive_index,indexes)
            get_feature(len(p_seq),pssm_dic,spd_file_path,positive_index,indexes)
            print(seq_number)
            seq_number+=1

        # if line is empty
        # end of file is reached
        if not line:
            break
        # print("Line{}: {}".format(count, line.strip()))
file = open('dataset-bphos/p{}.txt'.format(ptm_site), 'r')
get_indexes(ptm_site)
import numpy as np
import pandas as pd
#np.savez('unbalanced_S_test.npz',X_train,y_train)
np.savez('ml-window-comparision/imbalanced_dataset/unbalanced_{}_{}.npz'.format(ptm_site,WINDOW_SIZE), X_train, y_train)

pos_count = 0
neg_count = 0
for i, val in enumerate(y_train):
    if val == 1:
        pos_count += 1
    else : neg_count += 1
print(pos_count)