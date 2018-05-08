import pandas as pd

from config.config import *

activity_filename = "./data/2017_01_12.csv"
activity_product_filename = "./data/activity_product_2017_01_12.csv"

activity_product_df = pd.read_csv(activity_product_filename).astype("str")

print(activity_product_df.shape)
print("$#$#$#$#$#$#$#$")
print(activity_product_df.drop_duplicates().shape)

activity_price_df = activity_product_df[activity_product_df[activity_price]!="nan"]
print(activity_product_df.shape)

print(activity_price_df[start_time].min())

# def test(x):
#     if pd.isnull(x):
#         print("HAHHAHAHAHHSHSHASHDASHDASH")
#         print(x)
#     # if not x:
#     try:
#         y = int(x)
#         y = y + 1
#     except:
#         print("()()()()()()()()")
#         print(x)
#
#
#     return x
# print("#########")
# list(map(test,activity_price_df[danpin_id]))
# print("@@@@@@@@@@@@@@@@")
#
# print("活动商品id不为空")
# activity_products_id_df = activity_price_df[activity_price_df[activity_products_id]!="nan"]
# print(activity_products_id_df.shape)


#
# print("活动单品id不为空")
# danpin_id_df = activity_price_df[~pd.isnull(activity_price_df[danpin_id])]#(activity_price_df[danpin_id]!="nan") &~pd.isnull(activity_price_df[danpin_id])
# print(danpin_id_df.shape)

#
# print("活动商品id nan")
# activity_products_id_df_nan = activity_price_df[activity_price_df[activity_products_id]=="nan"]
# activity_products_id_df_nan.to_csv("./data/activity_products_id_df_nan.csv",index=False)

# danpin_id_df_nan = activity_price_df[pd.isna(activity_price_df[danpin_id])]
# print(danpin_id_df_nan.shape)
#
# tem = activity_price_df.append(danpin_id_df)
# tem = tem.drop_duplicates(subset=["fk_activity_id",
#                                   "fk_listing_id","activity_products_id","start_time","activity_price"],keep=False)
#
# print('#$#$#$#$#$#$%#$%#%$#%#%#%#%#%')
# print(tem.shape)
#tem.to_csv("./data/tem.csv",index=False)
# danpin_id_df_nan1 = activity_price_df[pd.isnull(activity_price_df[danpin_id])]#(activity_price_df[danpin_id]=="nan") &
# print(danpin_id_df_nan1.shape)


#danpin_id_df_nan.to_csv("./data/danpin_id_df_nan.csv",index=False)
#activity_price_df.to_csv("./data/activity_price.csv",index=False)
# df = pd.read_csv(filename,na_values='null').astype("str")
# result = df.groupby(activity_type).count()
# #统计各个类型活动的数量
# print(type(result))
# print(result)
#
# #统计单品类活动的城市ID，品牌ID，门店不为空的数量
# danpin_df = df[df[activity_type]=="6"]


#

# danpin_city_notnull = danpin_df[danpin_df[city_ids]!="nan"]
#

# danpin_brands = danpin_df[danpin_df[brand_ids]!="nan"]
#
# danpin_store = danpin_df[(danpin_df[store_ids]!="nan") & (danpin_df[city_ids]!="nan")]




# print("city not null")
# print(danpin_city_notnull.shape)
#
# print(" brands not null")
# print(danpin_brands.shape)
#
# print("store not null")
# print(danpin_store.shape)
# print("city")
# print(danpin_store.groupby(city_ids).count())
#
# print("store id")
# print(danpin_store.groupby(store_ids).count())
# danpin_df.to_csv("./data/danpin.csv",index=False)
#
# danpin_city_notnull.to_csv("./data/danpin_city_notnull.csv",index=False)
#
# danpin_city_brands.to_csv("./data/danpin_city_brands.csv",index=False)
#
# danpin_store.to_csv("./data/danpin_store.csv",index=False)