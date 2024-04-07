import itertools
import operator
from graphframes import *
import sys
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import os


if __name__ == "__main__":
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes: graphframes:0.8.2 - spark3.1 - s_2.12pyspark - shell"
    spark_context = SparkContext.getOrCreate()
    spark_context.setLogLevel("WARN")
    data = spark_context.textFile(sys.argv[2])
    header = data.first()
    data = data.filter(lambda row: row != header).map(lambda row: row.split(','))
    data = data.map(lambda row: (row[0], row[1]))
    filter_threshold = int(sys.argv[1])
    users_indexed = data.map(lambda row: row[0]).distinct().zipWithIndex().collectAsMap()
    reverse_index_users = {value: key for key, value in users_indexed.items()}
    business_indexed = data.map(lambda row: row[1]).distinct().zipWithIndex().collectAsMap()
    reverse_index_business = {value: key for key, value in business_indexed.items()}
    data_indexed = data.map(lambda row: (users_indexed[row[0]], business_indexed[row[1]]))
    data_user_business = data_indexed.map(lambda row: ((row[0]), [row[1]])).reduceByKey(operator.add).map(
        lambda row: (row[0], set(row[1]))).collectAsMap()
    total_users = users_indexed.values()
    users_pairs = spark_context.parallelize(list(itertools.combinations(total_users, 2)))
    users_common_business_rdd = users_pairs.map(
        lambda row: (row[0], row[1], set(data_user_business[row[0]] & data_user_business[row[1]]))).filter(
        lambda row: len(row[2]) >= filter_threshold).map(
        lambda row: tuple([reverse_index_users[row[0]], reverse_index_users[row[1]]]))
    nodes1 = users_common_business_rdd.map(lambda row: tuple([row[0]])).collect()
    nodes2 = users_common_business_rdd.map(lambda row: tuple([row[1]])).collect()
    nodes = spark_context.parallelize(list(set(nodes2 + nodes1)))
    spark_context_sql = SQLContext(spark_context)
    users_common_business_rdd1 = users_common_business_rdd.collect()
    users_common_business_rdd2 = users_common_business_rdd.map(lambda row: (row[1], row[0])).collect()
    vertices = spark_context_sql.createDataFrame(nodes, ['id'])
    edges = spark_context_sql.createDataFrame(users_common_business_rdd1 + users_common_business_rdd2, ['src', 'dst'])
    graph = GraphFrame(vertices, edges)
    value = graph.labelPropagation(maxIter=5)
    value = value.rdd.map(lambda row: (row[1], [row[0]])).reduceByKey(operator.add).map(
        lambda row: sorted(row[1])).sortBy(lambda row: (len(row), row)).collect()

    def write(value):
        for i in value:
            file.write(str(i).replace("[", "").replace("]", "") + '\n')
    file = open(sys.argv[3], 'w')
    write(value)
