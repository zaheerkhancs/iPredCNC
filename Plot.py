# region "Import Libraries"
from scipy import interp
# Avoiding warning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import bayes
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifiaer , \
    RandomForestClassifier , \
    AdaBoostClassifier , \
    GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score , \
    confusion_matrix , \
    roc_auc_score , \
    average_precision_score , \
    roc_curve , \
    f1_score , \
    recall_score , \
    matthews_corrcoef , \
    auc , cohen_kappa_score
# #############################################################################
# Data IO and generation

# Import some data to play with
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
# endregion

random_state = np.random.RandomState(0)
iRec = 'oftFeatures-001-02.csv'
D = pd.read_csv(iRec)  # , skiprows=1)#header=None)  # Using pandas
X = D.iloc[: , :-1].values
y = D.iloc[: , -1].values
# X = sklearn.preprocessing.scale(X)
X , y = shuffle(X , y)  # Avoiding bias
scale = StandardScaler()
y = LabelEncoder().fit_transform(y)
cv = StratifiedKFold(n_splits = 10)
# classifier = KNeighborsClassifier(n_neighbors=7)
# classifier = DecisionTreeClassifier()
# classifier = BaggingClassifier()
# classifier = RandomForestClassifier()
# classifier = AdaBoostClassifier(n_estimators=50,learning_rate=1,random_state=0)
# classifier = SVC(C=2.0, gamma=0.5,  cache_size=200, class_weight=None, kernel='linear', max_iter=-1, probability=True, random_state=None, shrinking=True,    tol=0.001, verbose=False),
# classifier = SVC(C=2.0,gamma=0.5,cache_size=200,class_weight=None,kernel='rbf',max_iter=-1,probability=True,random_state=None,shrinking=True,tol=0.001,verbose=False),
# classifier = SVC(kernel='linear', probability=True,random_state=random_state)
# classifier = svm.SVC(kernel='rbf', probability=True,random_state=random_state)
classifier = MLPClassifier()
# classifier = QuadraticDiscriminantAnalysis()
tprs = []
aucs = []
mean_fpr = np.linspace(0 , 1 , 100)
random_state = np.random.RandomState(0)
i = 0
for train , test in cv.split(X , y):
    probas_ = classifier.fit(X[train] , y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr , tpr , thresholds = roc_curve(y[test] , probas_[: , 1])
    tprs.append(interp(mean_fpr , fpr , tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr , tpr)
    aucs.append(roc_auc)
    plt.plot(fpr , tpr , lw = 1 , alpha = 0.3 ,
             label = '      ROC over fold %d (AUC = %0.2f)' % (i , roc_auc))

    i += 1
plt.plot([0 , 1] , [0 , 1] , linestyle = '--' , lw = 2 , color = 'r' ,
         label = 'Chance' , alpha = .8)

mean_tpr = np.mean(tprs , axis = 0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr , mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr , mean_tpr , color = 'b' ,
         label = r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc , std_auc) ,
         lw = 2 , alpha = .8)

std_tpr = np.std(tprs , axis = 0)
tprs_upper = np.minimum(mean_tpr + std_tpr , 1)
tprs_lower = np.maximum(mean_tpr - std_tpr , 0)
plt.fill_between(mean_fpr , tprs_lower , tprs_upper , color = 'grey' , alpha = .2 ,
                 label = r'   $\pm$ 1 std. dev.')

plt.xlim([-0.05 , 1.05])
plt.ylim([-0.05 , 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve over  ' + str(classifier.__class__.__name__))
plt.legend(loc = "lower right")
plt.savefig('convGraphs/' + str(classifier.__class__.__name__ + '_roc.png') , dpi = 300)
plt.show()

print("done")
