import sys
import pyspark
from pyspark import SparkContext, SparkConf
import json

if __name__ == "__main__":
    spark_context = SparkContext.getOrCreate()
    spark_context.setLogLevel("WARN")

    data = spark_context.textFile(sys.argv[1]).map(lambda review: json.loads(review))

    task1 = {}

    total_number_of_reviews = data.filter(lambda row: row['review_id']).count()
    number_of_reviews_in_2018 = data.filter(lambda row: row['date'] and '2018' in row['date']).count()

    number_of_distinct_users = data.map(lambda row: row['user_id']).distinct().count()
    top10_user = data.map(lambda row: (row['user_id'], 1)).reduceByKey(lambda a, b: a + b).takeOrdered(10, lambda x: (-1 * x[1], x[0]))

    n_business = data.map(lambda row: row['business_id']).distinct().count()
    top10_business = data.map(lambda row: (row['business_id'], 1)).reduceByKey(lambda a, b: a + b).takeOrdered(10, lambda x: (-1 * x[1], x[0]))

    task1 = {
        'n_review': total_number_of_reviews,
        'n_review_2018': number_of_reviews_in_2018,
        'n_user': number_of_distinct_users,
        'top10_user': top10_user,
        'n_business': n_business,
        'top10_business': top10_business}

    json_object = json.dumps(task1)

    with open(sys.argv[2], "w") as file:
        file.write(json_object)
