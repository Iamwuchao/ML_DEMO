from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

x_train,y_train = load_svmlight_file("./train_data_process_more")
print(x_train.shape)
x_train_new,y_train_new = shuffle(x_train,y_train)
x_test,y_test = load_svmlight_file("./test_data_process_more")

lrm = LogisticRegression(penalty='l1',
                         C=1,
                         solver='saga',
                         verbose=1,
                         max_iter=100,
                         n_jobs=4
                         )
lrm.fit(x_train_new,y_train_new)
y_pred = lrm.predict(x_test)
auc = metrics.roc_auc_score(y_test,y_pred)
print(auc)