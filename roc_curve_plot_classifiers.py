import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pickle
import numpy as np
from itertools import cycle


def plot_roc_curves(fpr, tpr, roc_auc, label,protein_site):
    plt.figure()
    lw = 1.0
    colors = ['aqua', 'darkorange', 'cornflowerblue','crimson','darkgreen']
    for i in range(len(label)):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw, label='p{0} {1}(AUC = {2:0.2f})'.format(protein_site,label[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    # plt.figure(figsize=(30,10))
    plt.savefig('graphs/roc_curve_final'+protein_site+'.png', dpi=500)
    plt.show()



protein_site = 'Y'

npzfile = np.load('F:/ResearchProject/bioinformatics/ml-ten-bigram-nearmiss/dataset/nearmiss_balanced_{}.npz'.format(protein_site), allow_pickle=True)
x_train = npzfile['arr_0']
y_train = npzfile['arr_1']
# npzfile_test = np.load('balanced_bigram_profile/nearmiss_balanced_S_test_pssm_spd.npz', allow_pickle=True)
x_test = npzfile['arr_2']
y_test = npzfile['arr_3']
# x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.1, shuffle=True, random_state=47)


fpr = dict()
tpr = dict()
roc_auc = dict()
label = ('rof', 'rf', 'gb', 'adaboost', 'svm')


for i in range(len(label)):
    model_path = 'trained_model/final_pT_'+label[i]+'.pkl'
    modelfile = open(model_path, 'rb')
    model = pickle.load(modelfile)
    y_test_predict = model.predict(x_test)
    y_test_predict_prob = model.predict_proba(x_test)
    y_pred = y_test_predict_prob[:, 1]
    # Compute ROC curve and ROC area for each model
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test, y_pred)
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
plot_roc_curves(fpr, tpr, roc_auc, label,protein_site)
