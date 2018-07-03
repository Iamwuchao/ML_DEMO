import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
from sklearn.ensemble import RandomForestRegressor

train_file = "train.csv"
test_file = "test.csv"
feature_list = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
# pclass:等级  sibsp:兄弟姐妹  parch:of parents / children aboard the Titanic
#Cabin 船舱号


X_feature = ['Pclass','Name','Sex','Age','SibSp','Parch','Fare','Embarked']


def predict_age(data):
    age_df = data[['Age','Pclass','Name','Sex','SibSp','Parch','Fare']]
    know_age = age_df[age_df.Age.notnull()]
    know_age.fillna(0,inplace=True)
    know_age = know_age.values
    unknow_age = age_df[age_df.Age.isnull()].values
    y = know_age[:,0]
    x = know_age[:,1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
    rfr.fit(x,y)
    preT = rfr.predict(x)
    # mse = sklearn.metrics.mean_squared_error(y, preT)
    # print("####### train")
    # print(mse)
    # submission = pd.DataFrame({"preT": preT, "Age": y})
    # submission.to_csv("age.csv", index=False)

    predictsAges = rfr.predict(unknow_age[:,1::])
    data.loc[(data.Age.isnull()),'Age'] = predictsAges



def to_index(data,field):
    value_to_index = {}
    values = data.loc[:,field]
    index = 0
    for v in values:
        if(v not in value_to_index):
            value_to_index[v] = index
            index+=1
    data[field] = data[field].apply(lambda x:value_to_index[x])
    return data

def process_file(file):
    df = pd.read_csv(file,dtype={'Pclass':np.float32,'Name':str,"Sex":np.object,"Age":np.float32,"SibSp":np.float32,
                                 "Parch":np.float32,"Fare":np.float32,"Cabin":np.object,"Embarked":np.object
                                 })
    #print(df.shape)
    # feature_df = df.loc[:,feature_list]
    #print(feature_df.shape)
    return df

def process_name(name):
    if name is not None:
        str_arr = name.split(',')
        str_arr = str_arr[1].split('.')
        return str_arr[0]
    return ""

def xgb_test():
    train_data = process_file(train_file)
    test_data = process_file(test_file)
    data = pd.concat([train_data, test_data], ignore_index = True)
    x = data.loc[:,X_feature]
    y = data.loc[:,['Survived']]


    #处理性别
    x = to_index(x,'Sex')
    print(x.dtypes)

    #处理名字
    x['Name'] = x['Name'].apply(process_name)
    x = to_index(x,'Name')

    #处理Embarked
    x = to_index(x,'Embarked')

    #年龄处理
    predict_age(x)
    #print(x[x.Age.isnull()].shape)

    #缺失值处理
    x.fillna(0,inplace=True)
    train_x = x.iloc[0:891,:]
    train_y = y.iloc[0:891,:]
    print(train_x.shape)
    print(train_y.shape)

    test_x = x.iloc[891:,:]
    test_y = y.iloc[891:,:]
    print(test_x.shape)
    print(test_y.shape)

    #将X,Y处理为ndarray
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    dtrain = xgb.DMatrix(data=train_x, label=train_y)

    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'logloss',
              'gamma': 0.1,
              'min_child_weight': 1.1,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.01,
              'tree_method': 'exact',
              'seed': 0,
              'nthread': 12
              }

    num_round = 200
    bst = xgb.train(params, dtrain, num_boost_round=2000)
    preds = bst.predict(xgb.DMatrix(train_x))
    acc_val = sklearn.metrics.accuracy_score(train_y, preds.round())
    print("####### train")
    print(acc_val)

    test_p = bst.predict(xgb.DMatrix(np.array(test_x)))
    PassengerIds = test_data['PassengerId']
    test_p = test_p.round()
    submission = pd.DataFrame({"PassengerId": PassengerIds, "Survived": test_p.astype(np.int32)})
    submission.to_csv("submission.csv", index=False)

xgb_test()
