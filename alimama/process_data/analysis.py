'''
统计数据 了解数据吧
多少用户，
多少商品
多少店铺
'''

import pandas as pd
from conf.conf import *
df = pd.read_csv(train_file_csv)
print(df.shape)
# df1 = df[user_id].value_counts()
# print(df1.shape)
user_df= df[[user_id,item_id]].groupby(user_id).agg(['mean','count'])
print(user_df)
print(user_df.shape)