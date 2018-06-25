import pandas as pd

train_file = "train.csv"
test_file = "test.csv"

feature_list = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']

def process_data(file):
    df = pd.read_csv(file)
    #print(df.shape)
    feature_df = df.loc[:,feature_list]
    #print(feature_df.shape)
    return feature_df


#process_data(train_file)