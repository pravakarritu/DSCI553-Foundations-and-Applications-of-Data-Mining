import math
import operator
import sys
import time
import pyspark
from pyspark import SparkContext, SparkConf
import itertools
from functools import reduce

if __name__ == "__main__":
    spark_context = SparkContext.getOrCreate()
    spark_context.setLogLevel("WARN")
    start_time = time.time()
    def apriori(iterator):
        data_val = []

        for i in iterator:
            data_val.append(i)
        support = (len(data_val) / total_length) * support_value
        items = set()
        for i in data_val:
            for j in i[1]:
                items.add(j)

        candidate = []
        for i in items:
            count = 0
            for j in data_val:
                if i in j[1]:
                    count += 1
            if count >= support:
                candidate.append(i)

        candidate_list = list()
        candidate_list.append(candidate)
        number_of_items = 2
        item_set = candidate
        items = list(itertools.combinations(candidate, 2))

        while len(item_set):
            candidate = []
            item_set = set()
            for i in items:
                count = 0
                for j in data_val:
                    if set(i).issubset(j[1]):
                        count += 1
                if count >= support:
                    candidate.append(i)
                    for val in i:
                        item_set.add(val)
            number_of_items = number_of_items + 1
            if len(candidate):
                candidate_list.append(candidate)
            items = list(itertools.combinations(list(item_set), number_of_items))
        candidate_list = (reduce(operator.concat, candidate_list))
        new_list=[]
        for i in range(len(candidate_list)):
            new_list.append([candidate_list[i],1])
        return new_list


    def count_candidates(iterator):
        final_list = []
        data_val = []
        for i in iterator:
            data_val.append(i)
        for i in candidates:
            count = 0
            for j in data_val:
                if 'tuple' in str(type(i)):
                    if set(i).issubset(set(j[1])):
                        count += 1
                else:
                    if i in j[1]:
                        count += 1
            final_list.append([i, count])
        return final_list


    def write(item_list):
        for i in range(len(item_list)):
            if not 'tuple' in str(type(item_list[i])):
                item_list[i] = [item_list[i]]
            else:
                item_list[i] = list(sorted(list(item_list[i])))
        item_list = sorted(item_list, key=(lambda x:(len(x),x)))


        current_length = 1
        items = []
        for rdd in item_list:
            new_length = len(rdd)
            if current_length == new_length:
                if rdd not in items:
                    items.append(rdd)
            else:
                if current_length == 1:
                    f.write(','.join("('{}')".format(i[0]) for i in items) + '\n\n')
                    items = []
                else:
                    f.write(','.join(str(tuple(i)) for i in items) + '\n\n')
                    items= []
                current_length += 1
                if rdd not in items:
                    items.append(rdd)

        if current_length == 1:
            f.write(','.join('({})'.format(i[0]) for i in items) + '\n\n')
        else:
            f.write(','.join(str(tuple(i)) for i in items) + '\n\n')



    data = spark_context.textFile(sys.argv[3])
    header = data.first()
    data = data.filter(lambda row: row != header).map(lambda row: row.split(','))
    support_value = int(sys.argv[2])
    case_number = int(sys.argv[1])

    if case_number == 1:
        data = data.map(lambda row: (row[0], [row[1]])).reduceByKey(operator.add).map(lambda row: (row[0], list(set(row[1]))))
    else:
        data = data.map(lambda row: (row[1], [row[0]])).reduceByKey(operator.add).map(lambda row: (row[0], list(set(row[1]))))

    total_length = len(data.collect())
    candidates = data.mapPartitions(apriori).reduceByKey(operator.add).filter(lambda x : x[1]>=1).map(lambda x:x[0]).distinct().collect()
    frequent_itemsets = data.mapPartitions(count_candidates).reduceByKey(operator.add).filter(lambda x: x[1] >= support_value).map(lambda x: x[0]).distinct().collect()
    f = open(sys.argv[4], 'w')
    f.write('Candidates:' + '\n')
    write(candidates)
    f.write('Frequent Itemsets:' + '\n')
    write(frequent_itemsets)
    f.close()

    end_time=time.time()
    print("Duration:{}".format(end_time-start_time))
