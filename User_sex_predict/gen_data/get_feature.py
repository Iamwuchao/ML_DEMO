'''
特征的合并
'''
import pandas as pd
from conf.conf import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn

def merge_feature():
    files = [user_browse_count_file,user_browse_kinds_file,
             user_browse_pre_buy_file,user_buy_items_file,
             user_female_brand_file,user_collect_count_file,
             user_collect_kinds_file,user_items_types_file]
    dfs = []
    result_df = pd.read_csv(user_sex_activity_file)
    female = result_df[result_df["sex"]>0]
    print("@@@@@@@")
    print(result_df.shape)
    print(female.shape)
    for file_name in files:
        df = pd.read_csv(file_name)
        dfs.append(df)
    for df in dfs:
        result_df = pd.merge(result_df,df,how="left",on='member_id')
    print(result_df.shape)
    result_df = result_df.fillna(0)
    Y = result_df[["sex"]]
    result_df.drop('sex',axis=1,inplace=True)
    result_df.drop('member_id',axis=1,inplace=True)
    X=result_df
    X=X.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state=0)
    lr_model = LogisticRegression(C=1.0, class_weight="balanced")
    lr_model.fit(X_train, y_train)
    pred_y = lr_model.predict(X_test)
    male = pred_y[pred_y>0]
    print(pred_y.shape)
    print(male.shape)

    test_male = y_test[y_test>0]
    print(test_male.shape)

    acc_val = sklearn.metrics.accuracy_score(y_test, pred_y)
    print("#######")
    print(acc_val)


merge_feature()