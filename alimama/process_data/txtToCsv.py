'''
将文本文件转换为csv文件
'''
from conf.conf import *
import csv
def processData():
    # train_header = "instance_id item_id item_category_list item_property_list item_brand_id item_city_id item_price_level\
    #     item_sales_level item_collected_level item_pv_level user_id user_gender_id user_age_level user_occupation_id\
    #     user_star_level context_id context_timestamp context_page_id predict_category_property shop_id\
    #     shop_review_num_level shop_review_positive_rate shop_star_level shop_score_service shop_score_delivery\
    #     shop_score_description is_trade"
    test_header = "instance_id item_id item_category_list item_property_list item_brand_id item_city_id " \
                  "item_price_level item_sales_level item_collected_level item_pv_level user_id user_gender_id" \
                  " user_age_level user_occupation_id user_star_level context_id context_timestamp context_page_id " \
                  "predict_category_property shop_id shop_review_num_level shop_review_positive_rate shop_star_level shop_score_service " \
                  "shop_score_delivery shop_score_description"

    index_names = test_header.split()
    print(len(index_names))
    for str in index_names:
        print(str)
    print(type(index_names))
    with open(train_file_csv,'w') as csvfile:
        writer = csv.writer(csvfile,delimiter=',',quotechar='"')
        #writer.writerow(index_names)
        i = 0
        with open(train_file_txt) as file:
            for line in file:
                row_data = line.split()
                writer.writerow(row_data)
                i+=1
                if i>100:
                    csvfile.flush()
                    i=0
        csvfile.flush()







processData()
