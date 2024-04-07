import math
import operator
import sys
import time
import pyspark
from pyspark import SparkContext, SparkConf
import json
import xgboost as xgb
import pandas as pd
import csv
import pickle
import os
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
            yelp_train_data = yelp_train_file.filter(lambda row: row != header_train_file).map(lambda row: row.split(','))
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

    def write_output(data_write):
        for data_val in data_write:
            writer.writerow(data_val)
    file3 = open("data1.csv", 'w')
    writer = csv.writer(file3)
    writer.writerow(['user_id', 'business_id', 'business_average_stars', 'business_review_count', 'user_average_stars', 'user_review_count', 'train_stars'])
    write_output(data1.collect())
    file3.close()
    data = pd.read_csv("data1.csv")
    to_predict = data.train_stars.values
    parameters = data.drop(['user_id', 'business_id', 'train_stars'], axis=1).values
    model = xgb.XGBRegressor(n_estimators=500, subsample=0.9, colsample_bytree=0.9, learning_rate=0.1)
    model = model.fit(parameters, to_predict)
    pickle.dump(model, open("train.model", 'wb'))
    test_file = spark_context.textFile(sys.argv[2])
    header_test_file = test_file.first()
    test_file_data = test_file.filter(lambda row: row != header_train_file).map(lambda row: row.split(','))
    test_data = test_file_data.map(lambda row: (row[1], (row[0])))
    train_data = yelp_train_data.map(lambda row: (row[1], (row[0], row[2])))
    user_data2 = user_data.map(lambda row: (row['user_id'], (row['average_stars'], row['review_count'], row['yelping_since'])))
    business_data1 = business_data.map(lambda row: (row['business_id'], (row['stars'], row['review_count'])))
    data3 = test_data.join(business_data1)
    data2 = data3.map(lambda row: (row[1][0], (row[0], row[1][1][0], row[1][1][1]))).join(user_data2)
    data8 = data2.map(lambda row: (row[0],  row[1][0][0], row[1][0][1], row[1][0][2], row[1][1][0], row[1][1][1]))
    file1 = open("test.csv", 'w')
    writer = csv.writer(file1)
    writer.writerow(['user_id', 'business_id', 'business_average_stars', 'business_review_count', 'user_average_stars', 'user_review_count'])
    write_output(data8.collect())
    file1.close()
    test = pd.read_csv("test.csv")
    user_id, business_id = test.user_id.values, test.business_id.values
    output = pickle.load(open("train.model", 'rb')).predict(test.drop(['user_id', 'business_id'], axis=1).values)
    output1 = pd.DataFrame()
    output1["user_id"], output1["business_id"],output1["prediction"] = user_id, business_id, output
    file2 = open(sys.argv[3], 'w')
    writer = csv.writer(file2)
    writer.writerow(['user_id', 'business_id', 'prediction'])
    output_data = spark_context.parallelize([[str(value.user_id), str(value.business_id), str(value.prediction)] for _, value in output1.iterrows()])
    output_data_rdd = output_data.collect()
    for i in output_data_rdd:
        file2.write(",".join(i)+'\n')
    file2.close()
    end_time = time.time()
    print("Duration:{}".format(end_time - start_time))




