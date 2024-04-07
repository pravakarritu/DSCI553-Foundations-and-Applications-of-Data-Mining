import math
import operator
import sys
import time
import pyspark
from pyspark import SparkContext, SparkConf
import os
import json
import xgboost as xgb
import pandas as pd
import csv
import pickle

if __name__ == "__main__":

    spark_context = SparkContext.getOrCreate()
    spark_context.setLogLevel("WARN")
    start_time = time.time()
    directory_path = sys.argv[1]
    files_path = [directory_path + '/' + x for x in os.listdir(directory_path)]
    business_data = ''
    review_train_data = ''
    yelp_train_data = ''
    yelp_val_data = ''
    user_data = ''
    for i in files_path:
        if 'business.json' in i:
            business_data = spark_context.textFile(i).map(lambda row: json.loads(row))
        elif 'user' in i:
            user_data = spark_context.textFile(i).map(lambda row: json.loads(row))
        elif 'review_train' in i:
            review_train_data = spark_context.textFile(i).map(lambda row: json.loads(row))
        elif 'yelp_train.csv' in i:
            yelp_train_file = spark_context.textFile(i)
            header_train_file = yelp_train_file.first()
            yelp_train_data = yelp_train_file.filter(lambda row: row != header_train_file).map(
                lambda row: row.split(','))
        elif 'yelp_val_in.csv' in i:
            yelp_val_file = spark_context.textFile(i)
            header_val_file = yelp_val_file.first()
            yelp_val_data = yelp_val_file.filter(lambda row: row != header_train_file).map(lambda row: row.split(','))
    train_data = yelp_train_data.map(lambda row: (row[1], (row[0], row[2])))
    user_data1 = user_data.map(lambda row: (row['user_id'], (row['average_stars'], row['review_count'], row['yelping_since'])))
    business_data1 = business_data.map(lambda row: (row['business_id'], (row['stars'], row['review_count'])))
    data3 = train_data.join(business_data1)
    data2 = data3.map(lambda row: (row[1][0][0], (row[0], row[1][0][1], row[1][1][0], row[1][1][1]))).join(user_data1)
    data1 = data2.map(lambda row: (row[0], row[1][0][0], row[1][0][2], row[1][0][3], row[1][1][0], row[1][1][1], float(row[1][0][1])))
    data = yelp_train_data.map(lambda row: (row[1], row[0], row[2]))
    data_user_list = data.map(lambda row: (row[1], [row[0]])).reduceByKey(operator.add)
    data4 = data_user_list.map(lambda row: (row[0], set(row[1]))).collectAsMap()
    users = data.map(lambda row: (row[0])).distinct().zipWithIndex().collectAsMap()
    index_users = {value: key for key, value in users.items()}
    business_dict = data.map(lambda row: (row[1])).distinct().zipWithIndex().collectAsMap()
    index_business = {value: key for key, value in business_dict.items()}
    new_rdd2 = data.map(lambda row: (business_dict[row[1]], [(users[row[0]], float(row[2]))])).reduceByKey(operator.add)
    new_rdd = new_rdd2.collect()
    num_rating_each_business = data.map(lambda row: (row[1], ([row[2]]))).reduceByKey(operator.add).map(lambda row: (row[0], len(row[1]))).collect()
    avg_rating_each_business = data.map(lambda row: (row[1], [float(row[2])])).reduceByKey(operator.add).map(lambda row: (row[0], (sum(row[1]) / len(row[1])))).collect()
    new_rdd2 = data.map(lambda row: (users[row[0]], [(business_dict[row[1]], float(row[2]))])).reduceByKey(operator.add)
    new_rdd1 = new_rdd2.collect()
    num_rating_each_user = data.map(lambda row: (row[0], ([row[2]]))).reduceByKey(operator.add).map(lambda row: (row[0], len(row[1]))).collectAsMap()
    avg_rating_each_user = data.map(lambda row: (row[0], [float(row[2])])).reduceByKey(operator.add).map(lambda row: (row[0], (sum(row[1]) / len(row[1])))).collectAsMap()

    def pearson_similarity(business1, business2):
        users1 = new_rdd[business1][1]
        users_item_1 = dict()
        for i in users1:
            users_item_1[i[0]] = i[1]
        l1 = len(users_item_1.keys())
        users2 = new_rdd[business2][1]
        users_item_2 = dict()
        for i in users2:
            users_item_2[i[0]] = i[1]
        l2 = len(users_item_2.keys())
        co_rated_users = set(users_item_1.keys()) & set(users_item_2.keys())
        numerator, denominator_1, denominator_2, denominator = 0.0, 0.0, 0.0, 0.0
        if len(co_rated_users):
            tem1_average_all_ratings1 = 0.0
            for k, v in users_item_1.items():
                if k in co_rated_users:
                    tem1_average_all_ratings1 += v
            item1_average_all_ratings = tem1_average_all_ratings1 / len(co_rated_users)
            item1_average_all_ratings2 = 0.0
            for k, v in users_item_2.items():
                if k in co_rated_users:
                    item1_average_all_ratings2 += v
            item2_average_all_ratings = item1_average_all_ratings2 / len(co_rated_users)
            for user in co_rated_users:
                numerator += float(users_item_1[user] - item1_average_all_ratings) * float(
                    users_item_2[user] - item2_average_all_ratings)
                denominator_1 += (users_item_1[user] - item1_average_all_ratings) ** 2
                denominator_2 += (users_item_2[user] - item2_average_all_ratings) ** 2
            denominator = math.sqrt(denominator_1) * math.sqrt(denominator_2)
            return (float(numerator) / float(denominator)) if numerator else 0.0
        if numerator == 0.0 or denominator == 0.0:
            return 0.0
        else:
            return float(numerator) / float(denominator)

    def predict(user, business_val):
        try:
            user = users[user]
            business_val = business_dict[business_val]
        except:
            return 0.0
        businesses = dict()
        num_businesses = new_rdd1[user][1]
        for i in num_businesses:
            businesses[i[0]] = i[1]
        numerator, denominator = 0.0, 0.0
        pearson_similarity_business = dict()
        for i in businesses.keys():
            pearson_similarity_business[i] = pearson_similarity(business_val, i)
        business_to_consider_list1 = {k: v for (k, v) in pearson_similarity_business.items() if v >= 0.0}
        business_to_consider_list3 = sorted(business_to_consider_list1.items(), key=(lambda x: -1 * x[1]))[:30]
        business_to_consider_list2 = sorted(business_to_consider_list1.items(), key=(lambda x: x[1]))[:1]
        business_to_consider_list = business_to_consider_list3 + business_to_consider_list2
        for i in business_to_consider_list:
            businesses1 = i[0]
            pearson_similarity_of_business = pearson_similarity_business[businesses1]
            rating_of_business = businesses[businesses1]
            numerator += float(pearson_similarity_of_business * rating_of_business)
            denominator += float(abs(pearson_similarity_of_business))
        if numerator == 0.0 or denominator == 0.0:
            businesses_rated_all_users1 = new_rdd[business_val][1]
            sum_rating1 = []
            for i in businesses_rated_all_users1:
                sum_rating1.append(i[1])
            val1 = float(sum(sum_rating1) / len(sum_rating1))
            businesses_rated_all_users = new_rdd1[user][1]
            sum_rating = []
            for i in businesses_rated_all_users:
                sum_rating.append(i[1])
            val2 = float(sum(sum_rating) / len(sum_rating))
            return float((val2 + val1) / 2)
        else:
            return float(numerator) / float(denominator)

    def write_output(data_write):
        for data_val in data_write:
            writer.writerow(data_val)
    file3 = open("data2.csv", 'w')
    writer = csv.writer(file3)
    writer.writerow(['user_id', 'business_id', 'business_average_stars', 'business_review_count', 'user_average_stars','user_review_count', 'train_stars'])
    write_output(data1.collect())
    file3.close()
    data = pd.read_csv("data2.csv")
    to_predict = data.train_stars.values
    parameters = data.drop(['user_id', 'business_id', 'train_stars'], axis=1).values
    model = xgb.XGBRegressor(n_estimators=500, subsample=0.9, colsample_bytree=0.9, learning_rate=0.1)
    model = model.fit(parameters, to_predict)
    pickle.dump(model, open("train2.model", 'wb'))
    test_file = spark_context.textFile(sys.argv[2])
    header_test_file = test_file.first()
    test_file_data = test_file.filter(lambda row: row != header_train_file).map(lambda row: row.split(','))
    test_data = test_file_data.map(lambda row: (row[1], (row[0])))
    train_data = yelp_train_data.map(lambda row: (row[1], (row[0], row[2])))
    user_data2 = user_data.map(lambda row: (row['user_id'], (row['average_stars'], row['review_count'], row['yelping_since'])))
    business_data1 = business_data.map(lambda row: (row['business_id'], (row['stars'], row['review_count'])))
    data3 = test_data.join(business_data1)
    data2 = data3.map(lambda row: (row[1][0], (row[0], row[1][1][0], row[1][1][1]))).join(user_data2)
    data8 = data2.map(lambda row: (row[0], row[1][0][0], row[1][0][1], row[1][0][2], row[1][1][0], row[1][1][1]))
    file1 = open("test2.csv", 'w')
    writer = csv.writer(file1)
    writer.writerow(['user_id', 'business_id', 'business_average_stars', 'business_review_count', 'user_average_stars','user_review_count'])
    write_output(data8.collect())
    file1.close()
    test = pd.read_csv("test2.csv")
    user_id, business_id = test.user_id.values, test.business_id.values
    output = pickle.load(open("train2.model", 'rb')).predict(test.drop(['user_id', 'business_id'], axis=1).values)
    output1 = pd.DataFrame()
    output1["user_id"], output1["business_id"], output1["prediction"] = user_id, business_id, output
    output_data = spark_context.parallelize([[str(value.user_id), str(value.business_id), str(value.prediction)] for _, value in output1.iterrows()])
    output_data_rdd_model = output_data.map(lambda row: ((row[0], row[1]), row[2]))
    test_data_item_based = spark_context.textFile(sys.argv[2])
    header1 = test_data_item_based.first()
    test_data_item_based = test_data_item_based.filter(lambda row: row != header1).map(lambda row: row.split(','))
    output_item_based_rdd = test_data_item_based.map(lambda row: (row[0], row[1])).map(lambda row: ((row[0], row[1]), predict(row[0], row[1])))
    alpha = 0.00001
    output_2 = output_item_based_rdd.join(output_data_rdd_model).map(lambda row: (row[0][0], row[0][1], float(row[1][0]) * alpha + float(row[1][1]) * (1 - alpha)))
    output_data_rdd = output_2.collect()
    file2 = open(sys.argv[3], 'w')
    writer = csv.writer(file2)
    writer.writerow(['user_id', 'business_id', 'prediction'])
    for i in output_data_rdd:
        file2.write(",".join(list(map(str, i))) + '\n')
    file2.close()
    end_time = time.time()
    print("Duration:{}".format(end_time - start_time))
