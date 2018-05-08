
import pandas as pd

from conf.conf import *
'''
处理数据
'''
# brand_sex_df = pd.read_csv(brand_sex_file)
# brand_set = set()
#
# for index,row in brand_sex_df.iterrows():
#     brand_set.add(row["brand_id"])

def process_user_sex(filename=org_user_sex_file):
    '''
    处理用户性别数据，男：0，女: 1
    :return:
    '''
    user_sex_df = pd.read_csv(filename)
    user_sex_df['sex'] = user_sex_df[['sex']].apply(lambda x:x-1)
    user_sex_df.to_csv(user_sex_file,index=False)

def process(row):
    global brand_set
    keys = ["top1", "top2", "top3", "top4", "top5", "top6"]
    score = 0
    for key in keys:
        for i in range(6):
            if row[keys[i]] in brand_set:
                score += (6 - i)
    return score


def process_user_brand_score(filename=user_brand_favor_file):
    '''
    用户对女性品牌的偏好
    :return: 
    '''
    user_brand_favor_df = pd.read_csv(filename)
    user_brand_favor_df[female_brand_score] = 0

    print(len(brand_set))
    user_brand_favor_df[female_brand_score]=user_brand_favor_df.apply(process,axis=1)
    df = user_brand_favor_df[[member_id,female_brand_score]]
    test = user_brand_favor_df[user_brand_favor_df[female_brand_score]>0]
    print(len(test))
    df.to_csv(user_female_brand_file,index=False)

    # pd.merge(user_brand_favor_df,)
    # user_brand_t1 = user_brand_favor_df[["member_id","top1"]]


if __name__ =="__main__":
    print("&&&&&")
    #process_user_brand_score()
    process_user_sex()
