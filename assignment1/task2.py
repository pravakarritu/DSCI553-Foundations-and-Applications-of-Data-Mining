import sys
import time
import pyspark
from pyspark import SparkContext, SparkConf
import json

if __name__ == "__main__":
    spark_context = SparkContext.getOrCreate()
    spark_context.setLogLevel("WARN")

    data = spark_context.textFile(sys.argv[1]).map(lambda review: json.loads(review))

    task2 = {}

    start_default = time.time()
    top10_business = data.map(lambda row: (row['business_id'], 1))
    num_partitions = top10_business.getNumPartitions()
    num_items_per_partition = top10_business.glom().map(len).collect()
    end_default = time.time()


    def partitioner(key):
        return hash(key) % 100


    start_custom = time.time()
    custom_partition_rdd = top10_business.partitionBy(int(sys.argv[3]), partitioner)
    custom_num_partitions = custom_partition_rdd.getNumPartitions()
    custom_num_items_per_partition = custom_partition_rdd.glom().map(len).collect()
    end_custom = time.time()

    task2 = {
        'default': {'n_partition': num_partitions, 'n_items': num_items_per_partition,
                    'exe_time': end_default - start_default},
        'customized': {'n_partition': custom_num_partitions, 'n_items': custom_num_items_per_partition,
                       'exe_time': end_custom - start_custom}}
    json_object = json.dumps(task2)

    with open(sys.argv[2], "w") as file:
        file.write(json_object)
