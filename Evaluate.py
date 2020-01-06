# Avoiding warning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, \
    RandomForestClassifier,  \
    AdaBoostClassifier,    \
    GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, \
    confusion_matrix, \
    roc_auc_score, \
    average_precision_score, \
    roc_curve, \
    f1_score, \
    recall_score, \
    matthews_corrcoef, \
    auc,cohen_kappa_score


def warn(*args, **kwargs): pass
warnings.warn = warn
#iRec = 'data/ConClfTest.csv'
#iRec = 'data/featurewout.csv'
iRec = 'oftFeatures - Copy.csv'
D = pd.read_csv(iRec, skiprows=1)#header=None)  # Using pandas
X = D.iloc[:, :-1].values
y = D.iloc[:, -1].values

X, y = shuffle(X, y)  # Avoiding bias
scale = StandardScaler()
y = LabelEncoder().fit_transform(y)

Names = ['KNN', 'DT', 'Bagging', 'RF', 'AdaBoost','SVM', 'ANN']

Classifiers = [
    KNeighborsClassifier(n_neighbors=5),
    DecisionTreeClassifier(),
    BaggingClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(n_estimators=50,
                         learning_rate=1,
                         random_state=0),
    #SVC(kernel='rbf', probability=True),
    SVC(C=2.0, gamma=0.5,  cache_size=200, class_weight=None, kernel='linear', max_iter=-1, probability=True, random_state=None, shrinking=True,    tol=0.001, verbose=False),
    MLPClassifier()
]

def runClassifiers():
    i = 0
    Results = []  # compare algorithms
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    for classifier, name in zip(Classifiers, Names):
        accuray = []
        auROC= []
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
        ], dtype=int)
        print(classifier.__class__.__name__)
        model = classifier
        for (train_index, test_index) in cv.split(X, y):
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            model.fit(X_train, y_train)
            # Calculate ROC Curve and Area the Curve
            y_proba = model.predict_proba(X_test)[:, 1]
            FPR, TPR, _ = roc_curve(y_test, y_proba)
            mean_TPR += np.interp(mean_FPR, FPR, TPR)
            mean_TPR[0] = 0.0
            roc_auc = auc(FPR, TPR)
            y_artificial = model.predict(X_test)
            auROC.append(roc_auc_score(y_test, y_proba))
            accuray.append(accuracy_score(y_pred=y_artificial, y_true=y_test))
            avePrecision.append(average_precision_score(y_test, y_proba)) # auPR
            F1_Score.append(f1_score(y_true=y_test, y_pred=y_artificial))
            MCC.append(matthews_corrcoef(y_true=y_test, y_pred=y_artificial))
            Recall.append(recall_score(y_true=y_test, y_pred=y_artificial))
            AUC.append(roc_auc)
            CM += confusion_matrix(y_pred=y_artificial, y_true=y_test)
        accuray = [_*100.0 for _ in accuray]
        Results.append(accuray)
        mean_TPR /= cv.get_n_splits(X, y)
        mean_TPR[-1] = 1.0
        mean_auc = auc(mean_FPR, mean_TPR)
        plt.plot(
            mean_FPR,
            mean_TPR,
            linestyle='-',
            label='{} ({:0.3f})'.format(name, mean_auc), lw=2.0)

        TN, FP, FN, TP = CM.ravel()
        print('Accuracy: {0:.4f} %'.format(np.mean(accuray)))
        print('Sensitivity (+): {0:.4f} %'.format(float(TP / (TP + FN))*100.0))
        print('Specificity (-): {0:.4f} %'.format(float(TN / (TN + FP))*100.0))
        print('auROC: {0:.6f}'.format(mean_auc))
        print('F1-score: {0:.4f}'.format(np.mean(F1_Score)))
        print('MCC: {0:.4f}'.format(np.mean(MCC)))
        print('Recall: {0:.4f}'.format(np.mean(Recall)))
        print("Kappa :", cohen_kappa_score(y_test, y_proba.round()))
        print('auPR: {0:.4f}'.format(np.mean(avePrecision)))  # average_Precision

        #print('Confusion Matrix:')
        #print(CM)
        print('_______________________________________')
        '''
        result_.loc['Name'] = str(name)
        result_.loc['Acc'] = format(np.mean(accuray).round())
        result_.loc['AUC'] = format(mean_auc.round())
        result_.loc['F1'] = np.mean(F1_Score).round()
        result_.loc['MCC'] = format(np.mean(MCC).round())
        result_.loc['Sen'] = format(float(TP / (TP + FN))*100.0)
        result_.loc['Sp'] = format(float(TN / (TN + FP))*100.0)
        result_.loc['kappa'] = cohen_kappa_score(y_test, y_proba.round()).round()
        print(result_)
        '''

    #auROCplot()

def boxPlot(Results, Names):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.grid(True)
    plt.boxplot(Results, patch_artist=True, vert=True, whis=True, showbox=True)
    ax.set_xticklabels(Names)
    plt.xlabel('\nName of Classifiers')
    plt.ylabel('\nAccuracy (%)')
    plt.savefig('figure/AccuracyBoxPlot.png', dpi=100)
    plt.show()
def auROCplot():
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Random')
    plt.xlim([0.0, 1.00])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.savefig('figure/auROC.png', dpi=100)
    plt.show()
if __name__ == '__main__':
    runClassifiers()


