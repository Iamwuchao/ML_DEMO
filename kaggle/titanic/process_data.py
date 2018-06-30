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
list = []
with open("requests",'r') as f:
    index = 0
    for line in f.readlines():
        if(index<2):
            index+=1
            continue
        else:
            strarr = line.split()
            if(strarr[0] !=""):
                print(strarr[0])
                list.append(strarr[0]+"\n")
        index+=1
with open("list.txt","w") as f1:
    f1.writelines(list)

