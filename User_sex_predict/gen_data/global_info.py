import pandas as pd
from conf.conf import *

'''
统计数据信息
'''
def user_brand_count():
    '''
    统计一下用户偏好的品牌有多少种
    :return:
    '''
    ubf = pd.read_csv(user_brand_score_file)
    sum = ubf.groupby("brand_id").count()
    print("*****")
    print(sum)
    print(len(sum))
    bs = pd.read_csv(brand_sex_file)
    result = pd.merge(ubf,bs,on="brand_id")
    result.groupby("brand_id").count()
    print(len(result))

def user_brand_score():
    user_brand = pd.read_csv(user_brand_score_file)
    user_sex = pd.read_csv(user_sex_file)

user_brand_count()