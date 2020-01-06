from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, \
    GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.utils import shuffle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import accuracy_score, \
    confusion_matrix, \
    average_precision_score, \
    roc_curve, \
    f1_score, \
    matthews_corrcoef, \
    auc, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

# iRec = 'data/ConClfTest.csv'
# iRec = 'data/featurewout.csv'
iRec = 'oftFeatures - Copy.csv'
#iRec = 'oftFeatures-001-02.csv'
D = pd.read_csv(iRec, skiprows = 1)  # header=None)  # Using pandas
X = D.iloc[:, :-1].values
y = D.iloc[:, -1].values

X, y = shuffle(X, y)  # Avoiding bias
scale = StandardScaler()
y = LabelEncoder().fit_transform(y)

F = open('OtherClassifierResults.txt', 'w')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
random_state = np.random.RandomState(0)
i = 0
accuray = []
auROC = []
avePrecision = []
F1_Score = []
AUC = []
MCC = []
Recall = []
mean_TPR = 0.0
mean_FPR = np.linspace(0, 1, 100)
CM = np.array([[0, 0], [0, 0], ], dtype = int)

#F.write('Feature Description:' + str('optFeatures') + '\n\n')
#F.write('Features, Shape:' + str(np.asarray(X).shape) + '\n\n')

Names = ['Bagging', 'DT', 'ExtraTreeClassifier', 'ExtraTreesClassifier', 'GaussianNB', 'KNN', 'LDA', 'LR', 'QDA']

Classifiers = [
    BaggingClassifier(),
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    ExtraTreesClassifier(),
    GaussianNB(),
    GradientBoostingClassifier(),
    KNeighborsClassifier(),
    LinearDiscriminantAnalysis(),
    LogisticRegression(),
    QuadraticDiscriminantAnalysis(),
]
Results = []  # compare algorithms

for classifier, name in zip(Classifiers, Names):
    accuray = []
    auROC = []
    avePrecision = []
    F1_Score = []
    AUC = []
    MCC = []
    Recall = []
    mean_TPR = 0.0
    mean_FPR = np.linspace(0, 1, 100)
    CM = np.array([
        [0, 0],
        [0, 0],
    ], dtype = int)

    print(classifier.__class__.__name__)
    cv = StratifiedKFold(n_splits = 10, shuffle = True)
    F.write('Classifier Inuse:' + str(classifier.__class__.__name__) + '\n\n')
    F.write(
        "Fold" + "\t" + "ACC" + "\t\t" + "Sn" + "\t\t" + "Sp" + "\t\t" + "F-Score" + "\t\t" + "MCC" + "\t\t" + "Kappa" + "\t\t" + "APR" + "\t\t" + "ROC" + "\n\n")
    fold = 1
    for train, test in cv.split(X, y):
        Start_Time = time.time()
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]
        classifier.fit(X_train, y_train)
        # Calculate ROC Curve and Area the Curve
        y_proba = classifier.predict_proba(X_test)[:, 1]
        FPR, TPR, _ = roc_curve(y_test, y_proba)
        # mean_TPR += np.interp(mean_FPR, FPR, TPR)
        # mean_TPR[0] = 0.0
        roc_auc = auc(FPR, TPR)
        # auROC.append(roc_auc_score(y_test, y_proba))
        accuray = accuracy_score(y_pred = y_proba.round(), y_true = y_test)
        avePrecision = average_precision_score(y_test, y_proba)  # auPR
        F1_Score = f1_score(y_true = y_test, y_pred = y_proba.round())
        MCC = matthews_corrcoef(y_true = y_test, y_pred = y_proba.round())
        CM = confusion_matrix(y_pred = y_proba.round(), y_true = y_test)
        TN, FP, FN, TP = CM.ravel()

        AAC = round(float(np.mean(accuray)) * 100.0, 3)
        Sn = round(float(TP / (TP + FN)) * 100.0, 3)
        Sp = round(float(TN / (TN + FP)) * 100.0, 3)
        F1 = round(np.mean(F1_Score), 4)
        MCC = round(np.mean(MCC), 4)
        Kappa = round(cohen_kappa_score(y_test, y_proba.round()), 4)
        APR = round(np.mean(avePrecision), 4)
        AUROC = round(roc_auc, 4)

        print("Fold Value           :", fold)
        print('Accuracy             : {0:.4f} %'.format(AAC))
        print('Sensitivity (+)      : {0:.4f} %'.format(Sn))
        print('Specificity (-)      : {0:.4f} %'.format(Sp))
        print('F1-score             : {0:.4f}'.format(F1))
        print('MCC                  : {0:.4f}'.format(MCC))
        print('Kappa                : {0:.4f}'.format(Kappa))
        print('auPR                 : {0:.4f}'.format(APR))  # average_Precision
        print('auROC                : {0:.6f}'.format(AUROC))
        print("\n\n")

        """Draw ROC curve"""
        probas_ = classifier.predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw = 1, alpha = 0.3)  # ,label=' Fold %d (AUC = %0.2f)' % (i, roc_auc))
        fold += 1

        F.write(
            str(fold) + "\t\t\t" + str(AAC) + "\t\t" + str(Sn) + "\t\t" + str(Sp) + "\t\t" + str(F1) + "\t\t" + str(
                MCC) + "\t\t" + str(Kappa) + "\t\t" + str(APR) + "\t\t" + str(AUROC) + "\n")
        #F.write('\n\n')
F.close()
'''
plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', label = 'Chance', alpha = .8)
mean_tpr = np.mean(tprs, axis = 0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color = 'b',
         label = r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw = 2, alpha = .8)
std_tpr = np.std(tprs, axis = 0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color = 'grey', alpha = .2, label = r'   $\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Clf: " + classifier.__class__.__name__, loc = 'left')
plt.title("ROC: " + "Opt Feature", loc = 'right')
plt.legend(loc = "lower right")
# plt.savefig(fdirLayer1+classifier.__class__.__name__ + MidFile + '.png', dpi = 900)
plt.close()
print('  Feature Evaluation Finished.')
litoc = time.time()
# print('Time Taken.%.3f' % ((litoc-litic)/60))
F.close()
'''