from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
import logging
import numpy as np

logging.basicConfig(filename='test.log', level=logging.INFO)

x_train,y_train = load_svmlight_file("./train_data_process_more")
print(x_train.shape)
x_train_new,y_train_new = shuffle(x_train,y_train)
#print(type(x_train_new))
print(type(x_train_new[0]))
print(type(x_train_new[0].toarray()))
test = x_train_new[0].toarray()
x_test,y_test = load_svmlight_file("./test_data_process_more")
clf = SGDClassifier(loss='log',penalty="l1",
                    learning_rate="optimal",n_jobs=4,tol=0.00001,max_iter=1000,verbose=1)
clf.fit(x_train_new,y_train_new)
y_pred = clf.predict(x_test)
auc = metrics.roc_auc_score(y_test,y_pred)
print(auc)
