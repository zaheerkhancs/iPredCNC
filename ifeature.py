# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 07:52:46 2018

@author: zaheer
"""
import keras
import numpy
import sklearn
from keras.optimizers import SGD
from sklearn.metrics import roc_curve,auc,accuracy_score
from sklearn.preprocessing import LabelEncoder,scale
import time
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve,auc,precision_recall_curve,f1_score,average_precision_score,confusion_matrix
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Dense,Dropout,LSTM,Embedding,Activation,Lambda,Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential


def letter_to_index(letter):
    _alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    return next((i for i,_letter in enumerate(_alphabet) if _letter == letter),None)


def load_data(seqfile,test_split=0.2,maxlen=500):
    print('Loading data...')
    # df = pd.read_csv(seqfile)
    df = seqfile
    df.columns = df.columns.str.lstrip()
    df.columns = df.columns.str.rstrip()
    df['sequence'] = df['sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    df = df.reindex(np.random.permutation(df.index))
    train_size = int(len(df) * (1 - test_split))
    X_train = df['sequence'].values[:train_size]

    y_train = np.array(df['target'].values[:train_size])
    X_test = np.array(df['sequence'].values[train_size:])

    y_test = np.array(df['target'].values[train_size:])
    print('Average train sequence length: {}'.format(np.mean(list(map(len,X_train)),dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len,X_test)),dtype=int)))
    #X_train = sklearn.preprocessing(X_train)
    #X_test = sklearn.preprocessing(X_test)
    return pd.get_dummies(X_train,maxlen=maxlen),y_train, pd.get_dummies(X_test,maxlen=maxlen),y_test
    #return pd.get_dummies(X_train), pd.get_dummies(y_train), pd.get_dummies(X_test),pd.get_dummies(y_test)


def dlbl(filename_,feature_):
    dataframe = pd.read_csv(filename_,header=0,sep=',')
    data = dataframe[dataframe.columns[0:int(feature_)]]
    label = dataframe['Label'].values.tolist()
    return data,label


def one_hot_encoding(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    encode_labels = encoder.transform(labels)
    return encode_labels


def baseline_model():
    dnsModel = Sequential()
    # initfn= glorot_uniform(seed=1)
    dnsModel.add(Dense(50,input_dim= 11,kernel_initializer='uniform',activation='relu'))
    dnsModel.add(Dense(51,kernel_initializer='uniform',activation='relu'))
    #dnsModel.add(Dropout(0.2,input_shape=(20,)))
    dnsModel.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    # dnsModel.add(Activation(tf.nn.softmax))
    dnsModel.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return dnsModel


def bsmodel():
    dnsModel = Sequential()
    # initfn= glorot_uniform(seed=1)
    dnsModel.add(Dense(50,input_dim=1214,kernel_initializer='uniform',activation='relu'))
    dnsModel.add(Dense(2556,kernel_initializer='uniform',activation='relu'))
    dnsModel.add(Dropout(0.2,input_shape=(20,)))
    dnsModel.add(Dense(2556,kernel_initializer='uniform',activation='relu'))
    dnsModel.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    # dnsModel.add(Activation(tf.nn.softmax))
    dnsModel.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return dnsModel


def Custom_baseline_model(lrrate):
    dnsModel = Sequential()
    Opt_ = SGD(lr=0.0,momentum=0.9,decay=0.0,nesterov=False)
    dnsModel.add(Dense(50,input_dim=83,kernel_initializer='uniform',activation='relu'))
    dnsModel.add(Dense(20,kernel_initializer='uniform',activation='relu'))
    dnsModel.add(Dropout(0.2,input_shape=(20,)))
    dnsModel.add(Dense(20,kernel_initializer='uniform',activation='relu'))
    dnsModel.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    dnsModel.compile(loss='binary_crossentropy',optimizer=Opt_,metrics=['accuracy'])
    return dnsModel


def step_ecay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epoch_drop = 10
    lrate = initial_lrate * numpy.math.pow(drop,numpy.math.floor((1 + epoch) / epoch_drop))
    return lrate


def getStart():
    starttime = time()
    print("Starting Time.",starttime)


def getStop():
    endtime = time()


def acc(y_test,dltpred):
    return accuracy_score(y_test,dltpred)


def plotme_precisionRecallCurve(ytest,yhat):
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    from sklearn.utils.fixes import signature
    precision,recall,_ = precision_recall_curve(ytest,yhat)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall,precision,color='b',alpha=0.2,
             where='post')
    plt.fill_between(recall,precision,alpha=0.2,color='b',**step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0,1.05])
    plt.xlim([0.0,1.0])
    plt.title('Precision-Recall curve:',(average_precision_score))
    plt.savefig('graphs/Precision_recall_Curve.png')


def PlotMe(history):
    plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train','Test'],loc='upper left')
    plt.savefig('graphs/Accuracy.png')
    plt.show()
    plt.clf()

    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='upper left')
    plt.savefig('graphs/loss.png')
    plt.show()
    plt.clf()


def ttl_plotme(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])  # RAISE ERROR
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'],loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])  # RAISE ERROR
    plt.title('Model Loss Evaluate')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'],loc='upper left')
    plt.show()
    plt.savefig('graphs/ttlPlot.png')


def GetROC_Curve(fpr,tpr,roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr,tpr,color='red',
             lw=lw,label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Positive Test ROC')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('graphs/RoC_Curve_Gen.png')


def CMCmapPlotme(cm):
    classes = ['Hot','Cold']
    plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i,j in numpy.itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),
                 horizontalalignment="center",
                 color="white" if cm[i,j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def perf_measure(ytest,y_hat):
    yhat = y_hat.round()
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(yhat)):
        if ytest[i] == y_hat[i] == 1:
            TP += 1
        if yhat[i] == 1 and ytest[i] != yhat[i]:
            FP += 1
        if ytest[i] == yhat[i] == 0:
            TN += 1
        if yhat[i] == 0 and ytest[i] != yhat[i]:
            FN += 1

    return (TP,FP,TN,FN)


def cleandf(datafram):
    return cldf
