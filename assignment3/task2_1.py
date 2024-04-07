import math
import operator
import sys
import time
import pyspark
from pyspark import SparkContext, SparkConf

if __name__ == "__main__":

    spark_context = SparkContext.getOrCreate()
    spark_context.setLogLevel("WARN")
    start_time = time.time()
    data = spark_context.textFile(sys.argv[1])
    header = data.first()
    data = data.filter(lambda row: row != header).map(lambda row: row.split(','))
    data = data.map(lambda row: (row[0], row[1], row[2]))
    data_user_list = data.map(lambda row: (row[1], [row[0]])).reduceByKey(operator.add)
    data1 = data_user_list.map(lambda row: (row[0], set(row[1]))).collectAsMap()
    users = data.map(lambda row: (row[0])).distinct().zipWithIndex().collectAsMap()
    index_users = {value: key for key, value in users.items()}
    business_dict = data.map(lambda row: (row[1])).distinct().zipWithIndex().collectAsMap()
    index_business = {value: key for key, value in business_dict.items()}
    new_rdd = data.map(lambda row: (business_dict[row[1]], [(users[row[0]], float(row[2]))])).reduceByKey(operator.add).collectAsMap()
    new_rdd1 = data.map(lambda row: (users[row[0]], [(business_dict[row[1]], float(row[2]))])).reduceByKey(operator.add).collectAsMap()


    def pearson_similarity(business1, business2):
        users1 = new_rdd[business1]
        users_item_1 = dict()
        for i in users1:
            users_item_1[i[0]] = i[1]
        l1 = len(users_item_1.keys())
        users2 = new_rdd[business2]
        users_item_2 = dict()
        for i in users2:
            users_item_2[i[0]] = i[1]
        l2 = len(users_item_2.keys())
        co_rated_users = set(users_item_1.keys()) & set(users_item_2.keys())

        numerator, denominator_1, denominator_2, denominator = 0.0, 0.0, 0.0, 0.0
        if len(co_rated_users)>12:
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
        num_businesses = new_rdd1[user]
        for i in num_businesses:
            businesses[i[0]] = i[1]
        numerator, denominator = 0.0, 0.0
        pearson_similarity_business = dict()
        for i in businesses.keys():
            pearson_similarity_business[i] = pearson_similarity(business_val, i)
        business_to_consider_list = sorted(pearson_similarity_business.items(), key=(lambda x: -1 * x[1]))[:15]

        for i in business_to_consider_list:
            businesses1 = i[0]
            pearson_similarity_of_business = pearson_similarity_business[businesses1]
            rating_of_business = businesses[businesses1]
            numerator += float(pearson_similarity_of_business * rating_of_business)
            denominator += float(abs(pearson_similarity_of_business))

        if numerator == 0.0 or denominator == 0.0:
            businesses_rated_all_users1 = new_rdd[business_val]
            sum_rating1 = []
            for i in businesses_rated_all_users1:
                sum_rating1.append(i[1])
            val1 = float(sum(sum_rating1) / len(sum_rating1))

            businesses_rated_all_users = new_rdd1[user]
            sum_rating = []
            for i in businesses_rated_all_users:
                sum_rating.append(i[1])
            val2 = float(sum(sum_rating) / len(sum_rating))
            return float((val2 + val1) / 2)
        else:
            return float(numerator) / float(denominator)


    test_data = spark_context.textFile(sys.argv[2])
    header1 = test_data.first()
    test_data = test_data.filter(lambda row: row != header1).map(lambda row: row.split(','))
    test_data = test_data.map(lambda row: (row[0], row[1])).map(lambda row: (row[0], row[1], predict(row[0], row[1])))
    write_test_data = test_data.collect()

    def write(test_data_values):
        for i in test_data_values:
            file.write(",".join(list(map(str, i))) + "\n")

    file = open(sys.argv[3], 'w')
    file.write('user_id, business_id, prediction' + '\n')
    write(write_test_data)
    file.close()
    end_time = time.time()
    print("Duration:{}".format(end_time - start_time))
