import sys
import time
import pyspark
from pyspark import SparkContext, SparkConf
import json
import operator

if __name__ == "__main__":
    spark_context = SparkContext.getOrCreate()
    spark_context.setLogLevel("WARN")

    loading_and_average_time_start = time.time()

    test_review = spark_context.textFile(sys.argv[1]).map(lambda review: json.loads(review)).map(lambda row: (row['business_id'], row['stars']))
    business = spark_context.textFile(sys.argv[2]).map(lambda review: json.loads(review)).map(lambda row: (row['business_id'], row['city']))
    joined_rdd = business.leftOuterJoin(test_review)

    sum_stars_rdd = joined_rdd.filter(lambda row: row[1][1] is not None).map(lambda row: row[1]).reduceByKey(operator.add)
    sum_count_rdd = joined_rdd.filter(lambda row: row[1][1] is not None).map(lambda row: row[1]).map(lambda row: (row[0], 1)).reduceByKey(operator.add)
    average_rdd = sum_stars_rdd.leftOuterJoin(sum_count_rdd).map(lambda row: (row[0], float(row[1][0] / row[1][1])))

    loading_and_average_end = time.time()

    final_loading_and_average_time = loading_and_average_end - loading_and_average_time_start

    average_rdd_count = average_rdd.count()
    sorted_rdd = average_rdd.takeOrdered(average_rdd_count, lambda x: (-1 * x[1], x[0]))

    f = open(sys.argv[3], 'w')
    f.write('city,stars' + '\n')
    for rdd in sorted_rdd:
        f.write(','.join(str(x) for x in rdd) + '\n')
    f.close()

    task = {}

    sorting_time_python_start = time.time()
    rdd_list = average_rdd.collect()
    python_sort = sorted(rdd_list, key=lambda x: (-1 * x[1], x[0]))
    print(python_sort[:10])
    sorting_time_python_end = time.time()

    count = average_rdd.count()
    sorting_time_spark_start = time.time()
    spark_sort = average_rdd.takeOrdered(count, lambda x: (-1 * x[1], x[0]))
    print(spark_sort[:10])
    sorting_time_spark_end = time.time()

    task = {
        'm1': sorting_time_python_end - sorting_time_python_start,
        'm2': sorting_time_spark_end - sorting_time_spark_start,
        'reason': 'Spark is faster than Python in the execution of large datasets. Spark is faster because it supports distributed processing. On smaller datasets, Python can be faster.'
    }

    json_object = json.dumps(task)

    with open(sys.argv[4], "w") as file:
        file.write(json_object)
