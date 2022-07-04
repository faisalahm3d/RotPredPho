from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import numpy as np
from sklearn.model_selection import train_test_split

def split_balance_nearmiss(npz_file):
    X = npz_file['arr_0']
    y = npz_file['arr_0']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42,stratify=y)

    pos = 0
    for i in range(len(y_train)):
        if (y_train[i] == 1):
            pos += 1
    nm = NearMiss({0:pos*3,1:pos})
    x_train, y_train = nm.fit_resample(x_train, y_train)
    np.savez('balanced_pssm_spd_profile/nearmiss3-1_balanced_10.npz', x_train, y_train, x_test, y_test)

def split_balance_smote(npz_file):
    X = npz_file['arr_0']
    y = npz_file['arr_0']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42,stratify=y)
    nm = SMOTE(random_state=2)
    x_train, y_train = nm.fit_resample(x_train, y_train)
    np.savez('balanced_pssm_spd_profile/smote_balanced_S_10.npz', x_train, y_train, x_test, y_test)



protein_site = 'S'
npzfile = np.load('../pssm_spd_profile/unbalanced_10_'+protein_site+'.npz', allow_pickle = True)


split_balance_nearmiss(npzfile)



