from rotation_forest import RotationForestClassifier
import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, roc_auc_score,f1_score
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, roc_auc_score, f1_score
from error_measurement import matthews_correlation, sensitivity, specificity, auc, f1_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


residue_name = 'S'
dataset_path = 'F:/ResearchProject/bioinformatics/ml-ten-bigram-nearmiss/dataset/nearmiss_balanced_{}.npz'.format(residue_name)


models = {
    'rof': RotationForestClassifier(n_estimators=100, random_state=47, verbose=4, n_jobs=-2),
    'adaboost': AdaBoostClassifier(n_estimators=100, random_state=47),
    'rf': RandomForestClassifier(n_estimators=100, bootstrap=False, random_state=47, verbose=1),
    'svm': SVC(C=1, kernel='poly', random_state=47, verbose=True, probability=True),
    'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=47)
}

def cross_val(x_train, y_train, x_test, y_test, folds, initial_model, model_name):

    skf = StratifiedKFold(n_splits=folds,shuffle=True, random_state=42)
    model = initial_model
    accuracy = []
    mcc = []
    precision = []
    roc_auc = []
    Sensitivity = []
    Specificity = []
    auc_score = []
    f1 = []
    score = []

    for x in range(1):
        for train_index, test_index in skf.split(x_train, y_train):
            X_train, X_test = x_train[train_index], x_train[test_index]
            Y_train, Y_test = y_train[train_index], y_train[test_index]
            model.fit(X_train, Y_train)
            y_predict = model.predict(X_test)
            score.append(model.score(X_test, Y_test))

            accuracy.append(accuracy_score(Y_test, y_predict))
            mcc.append(matthews_corrcoef(Y_test, y_predict))
            precision.append(precision_score(Y_test, y_predict))
            roc_auc.append(roc_auc_score(Y_test, y_predict))
            #auc_score.append(auc(Y_test, y_predict))
            f1.append(f1_score(Y_test, y_predict))
            Sensitivity.append(sensitivity(Y_test, y_predict))
            Specificity.append(specificity(Y_test, y_predict))

    with open('trained_model/final_p{}_{}.pkl'.format(residue_name,model_name), 'wb') as f:
        pickle.dump(model, f)

    y_test_predict = model.predict(x_test)
    y_test_predict_prob = model.predict_proba(x_test)

    res = "                        Experimental result with {} classifier\n".format(model_name)
    res += "              (window size 10 is used extract structural and evolutionary features)\n"
    res += "---------------------------------------------------------------------------------------------\n"
    res += "                   {} folds CV          independent test\n".format(folds)
    res += "---------------------------------------------------------------------------------------------\n"
    res += "Accuracy:      {0:0.5f}({1:0.5f})          {2:0.5f}\n".format(np.mean(accuracy), np.std(accuracy), accuracy_score(y_test, y_test_predict))
    res += "MCC:           {0:0.5f}({1:0.5f})          {2:0.5f}\n".format(np.mean(mcc), np.std(mcc), matthews_corrcoef(y_test, y_test_predict))
    res += "Precision:     {0:0.5f}({1:0.5f})          {2:0.5f}\n".format(np.mean(precision), np.std(precision), precision_score(y_test, y_test_predict))
    res += "Roc AUC :      {0:0.5f}({1:0.5f})          {2:0.5f}\n".format(np.mean(roc_auc), np.std(roc_auc), roc_auc_score(y_test, y_test_predict))
    res += "F1 score:      {0:0.5f}({1:0.5f})          {2:0.5f}\n".format(np.mean(f1), np.std(f1), f1_score(y_test, y_test_predict))
    res += "Sensitivity:   {0:0.5f}({1:0.5f})          {2:0.5f}\n".format(np.mean(Sensitivity), np.std(Sensitivity), sensitivity(y_test, y_test_predict))
    res += "Specificity:   {0:0.5f}({1:0.5f})          {2:0.5f}\n\n".format(np.mean(Specificity), np.std(Specificity), specificity(y_test, y_test_predict))
    with open('results/final_p{}.txt'.format(residue_name), 'a') as f:
        f.write(res)

if __name__ == '__main__':
    npzfile = np.load(dataset_path, allow_pickle=True)
    x_train = npzfile['arr_0']
    y_train = npzfile['arr_1']
    x_test = npzfile['arr_2']
    y_test = npzfile['arr_3']

    for model_name in models:
        #file_name = item+'_'+site_protein
        cross_val(x_train, y_train, x_test, y_test, 5, models[model_name], model_name)

