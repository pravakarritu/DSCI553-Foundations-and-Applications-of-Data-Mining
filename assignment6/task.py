import math
import operator
import sys
import time
import pyspark
from pyspark import SparkContext, SparkConf
from sklearn.cluster import KMeans

if __name__ == "__main__":
    spark_context = SparkContext.getOrCreate()
    spark_context.setLogLevel("WARN")

    data = spark_context.textFile(sys.argv[1])
    header = data.first()
    data = data.filter(lambda row: row != header).map(lambda row: row.split(','))
    data1, data2, data3, data4, data5 = data.map(
        lambda row: (row[0], row[1], [row[i] for i in range(2, len(row))])).randomSplit([0.2, 0.2, 0.2, 0.2, 0.2],11111)

    data_checking = list()
    output_list = []
    dimension = len(data1.map(lambda row: row[2]).collect()[0])

    ds_cluster_check = dict()
    data_check = data1.map(lambda row: row[2])
    data_checking.append(data_check.map(lambda row: str(row)).collect())
    data_check_list = data_check.collect()
    data_check_index = data_check.zipWithIndex().map(lambda row: (row[1], row[0])).collectAsMap()
    input_clusters = int(sys.argv[2])
    data_check_index_reverse = {str(value): key for key, value in data_check_index.items()}

    kmeans = KMeans(n_clusters=5 * input_clusters, random_state=1,n_init=10).fit(data_check_list)
    val = kmeans.labels_
    val_points = spark_context.parallelize(val).zipWithIndex().map(lambda row: ((row[0]), [row[1]])).reduceByKey(
        operator.add).collectAsMap()

    rs = []
    rs_list = []
    for key, value in val_points.items():
        if len(value) == 1:
            rs.append(value)
            data_check_list.remove(data_check_index[value[0]])
            rs_list.append(data_check_index[value[0]])  

    kmeans = KMeans(n_clusters=input_clusters, random_state=1,n_init=10).fit(data_check_list)
    val = kmeans.labels_
    val_points = spark_context.parallelize(val).zipWithIndex().map(lambda row: ((row[0]), [row[1]])).reduceByKey(
        operator.add).collectAsMap()
    ds_cluster_check_list = list()
    for i in range(len(data_check_list)):
        ds_cluster_check_list.append([str(data_check_list[i]), val[i]])

    ds = []

    cs_cluster_check_list=list()

    def initialize_ds(val_points):
        n_list = []
        sum_list = []
        sumsq = []
        for key, value in val_points.items():
            n = len(value)
            sum_list_1 = [0 for i in range(10)]
            sumsq_1 = [0 for i in range(10)]
            n_list.append(n)
            for i in value:
                point_details = data_check_index[i]
                for j in range(len(point_details)):
                    sum_list_1[j] += float(point_details[j])
                    sumsq_1[j] += (float(point_details[j]) * float(point_details[j]))
            sum_list.append(sum_list_1)
            sumsq.append(sumsq_1)

        return n_list, sum_list, sumsq


    def initialize_cs(val_points,rs_list):
        n_list = []
        sum_list = []
        sumsq = []
        for key, value in val_points.items():
            n = len(value)
            sum_list_1 = [0 for i in range(10)]
            sumsq_1 = [0 for i in range(10)]
            n_list.append(n)
            for i in value:
                point_details = rs_list[i]
                cs_cluster_check_list.append([str(point_details),-1])
                for j in range(len(point_details)):
                    sum_list_1[j] += float(point_details[j])
                    sumsq_1[j] += (float(point_details[j]) * float(point_details[j]))
            sum_list.append(sum_list_1)
            sumsq.append(sumsq_1)

        return n_list, sum_list, sumsq


    ds_n_list, ds_sum_list, ds_sumsq = initialize_ds(val_points)


    def k_means_on_rs(rs_list):
        k = 5 * input_clusters
        rs_new = []
        cs_points = dict()
        if len(rs_list)>1:
            if len(rs_list) < k:
                k = int(len(rs_list) / 2)
            kmeans = KMeans(n_clusters=k, random_state=1,n_init=10).fit(rs_list)
            val = kmeans.labels_
            cs_val_points = spark_context.parallelize(val).zipWithIndex().map(lambda row: ((row[0]), [row[1]])).reduceByKey(
                operator.add).collectAsMap()
            rs_new = []
            cs_points = dict()
            for key, value in cs_val_points.items():
                if len(value) == 1:
                    rs_new.append(rs_list[value[0]])
                else:
                    cs_points[key] = value
        return rs_new, cs_points


    rs_new, cs_points = k_means_on_rs(rs_list)
    cs_n_list, cs_sum_list, cs_sumsq = initialize_cs(cs_points,rs_list)


    def calculate_ds_centroid(ds_n_list, ds_sum_list, ds_sumsq):
        ds_centroid = []
        ds_standard_deviation = []
        for i in range(len(ds_n_list)):
            centroid_list = []
            standard_deviation_list = []
            for j in range(len(ds_sum_list[0])):
                centroid = ds_sum_list[i][j] / ds_n_list[i]
                variance = (ds_sumsq[i][j] / ds_n_list[i]) - (ds_sum_list[i][j] / ds_n_list[i]) ** 2
                standard_deviation = math.sqrt(variance)
                centroid_list.append(centroid)
                standard_deviation_list.append(standard_deviation)
            ds_centroid.append(centroid_list)
            ds_standard_deviation.append(standard_deviation_list)
        return ds_centroid, ds_standard_deviation


    ds_centroid, ds_standard_deviation = calculate_ds_centroid(ds_n_list, ds_sum_list, ds_sumsq)


    def calculate_cs_centroid(cs_n_list, cs_sum_list, cs_sumsq):
        cs_centroid = []
        cs_standard_deviation = []
        for i in range(len(cs_n_list)):
            centroid_list = []
            standard_deviation_list = []
            for j in range(len(cs_sum_list[0])):
                centroid = cs_sum_list[i][j] / cs_n_list[i]
                variance = (cs_sumsq[i][j] / cs_n_list[i]) - (cs_sum_list[i][j] / cs_n_list[i]) ** 2
                standard_deviation = math.sqrt(variance)
                centroid_list.append(centroid)
                standard_deviation_list.append(standard_deviation)
            cs_centroid.append(centroid_list)
            cs_standard_deviation.append(standard_deviation_list)
        return cs_centroid, cs_standard_deviation


    cs_centroid, cs_standard_deviation = calculate_cs_centroid(cs_n_list, cs_sum_list, cs_sumsq)

    def load_new_points(data,ds_centroid,ds_standard_deviation,ds_n_list,ds_sum_list,ds_sumsq,cs_n_list,cs_sum_list,cs_sumsq):
        data_check2 = data.map(lambda row: row[2])
        data_check_list2 = data_check.collect()
        data_checking.append(data_check2.map(lambda row: str(row)).collect())

        data_check_index2 = data_check2.zipWithIndex().map(lambda row: (row[1], row[0])).collectAsMap()
        new_points_cs = dict()
        new_points_ds = dict()
        rs_list_2=[]
        for key, value in data_check_index2.items():
            point = value
            min_distance = []
            distance_max = sys.maxsize
            for i in range(len(ds_centroid)):
                distance = 0
                for j in range(len(ds_centroid[i])):
                    distance += (((float(point[j]) - ds_centroid[i][j]) / (ds_standard_deviation[i][j])) ** 2)
                distance = math.sqrt(distance)
                if distance < distance_max:
                    min_distance = [i, distance]
                    distance_max = distance
            if min_distance[1] < 2 * math.sqrt(len(point)):
                if min_distance[0] in new_points_ds.keys():
                    new_points_ds[min_distance[0]].append(value)
                else:
                    new_points_ds[min_distance[0]] = [value]

            else:
                point = value
                min_distance = []
                distance_max = sys.maxsize
                for i in range(len(cs_centroid)):
                    distance = 0
                    for j in range(len(cs_centroid[i])):
                        distance += (((float(point[j]) - cs_centroid[i][j]) / (cs_standard_deviation[i][j])) ** 2)
                    distance = math.sqrt(distance)
                    if distance < distance_max:
                        min_distance = [i, distance]
                        distance_max = distance

                if min_distance[1] < 2 * math.sqrt(len(point)):
                    if min_distance[0] in new_points_cs.keys():
                        new_points_cs[min_distance[0]].append(value)
                    else:
                        new_points_cs[min_distance[0]] = [value]
                else:
                    rs_list_2.append(value)
        for i in range(len(ds_n_list)):
            for key, val in new_points_ds.items():
                if i == key:
                    for value in range(len(val)):
                        ds_cluster_check_list.append([str(val[value]), key])
                        for j in range(len(ds_sum_list[0])):
                            ds_sum_list[i][j] += float(val[value][j])
                            ds_sumsq[i][j] += (float(val[value][j]) * float(val[value][j]))
                        ds_n_list[i] += 1

        for i in range(len(cs_n_list)):
            for key, val in new_points_cs.items():
                if i == key:
                    for value in range(len(val)):
                        cs_cluster_check_list.append([str(val[value]),-1])
                        for j in range(len(cs_sum_list[0])):
                            cs_sum_list[i][j] += float(val[value][j])
                            cs_sumsq[i][j] += (float(val[value][j]) * float(val[value][j]))
                        cs_n_list[i] += 1


        return ds_n_list, ds_sum_list, ds_sumsq,rs_list_2,cs_n_list,cs_sum_list,cs_sumsq

    def mahanobolis_distance(centroid1,centroid2,standard_deviation2):
        val=0
        for i in range(len(centroid1)):
            val+=(((centroid1[i]-centroid2[i])/standard_deviation2[i])**2)
        return math.sqrt(val)
    def find_merge_cs_cluster(cs_n_list,cs_sum_list,cs_sumsq,cs_centroid,cs_standard_deviation,cs_n_list2,cs_sum_list2,cs_sumsq_2,cs_centroid_2,cs_standard_deviation_2):
        cs_total=cs_n_list+cs_n_list2
        cs_sum_total=cs_sum_list+cs_sum_list2
        cs_sumsq_total=cs_sumsq+cs_sumsq_2
        cs_centroid_total=cs_centroid+cs_centroid_2
        cs_standard_deviation_total=cs_standard_deviation+cs_standard_deviation_2
        clusters_to_merge=[]
        merged=[]
        for i in range(len(cs_total)):
            for j in range(len(cs_total)):
                if i!=j and i not in merged and j not in merged:
                    if list(set(cs_standard_deviation_total[j]))[0]!=0:
                        distance=mahanobolis_distance(cs_centroid_total[i],cs_centroid_total[j],cs_standard_deviation_total[j])
                        if distance< 2*math.sqrt(dimension):
                            clusters_to_merge.append([i,j])
                            merged.append(i)
                            merged.append(j)


        cs_total_new=[]
        cs_sum_total_new=[]
        cs_sumsq_total_new=[]
        cs_centroid_total_new=[]
        cs_standard_deviation_total_new=[]
        indexes_merged=[]
        for i in clusters_to_merge:
            cluster1=i[0]
            cluster2=i[1]
            indexes_merged.append(cluster1)
            indexes_merged.append(cluster2)
            n_new=cs_total[cluster1]+cs_total[cluster2]

            cs_total_new.append(n_new)

            sum_new = [0 for i in range(10)]
            for val in range(dimension):
                sum_new[val] = cs_sum_total[cluster1][val] + cs_sum_total[cluster2][val]

            cs_sum_total_new.append(sum_new)

            sumsq_new = [0 for i in range(10)]
            for val in range(dimension):
                sumsq_new[val] = cs_sumsq_total[cluster1][val] + cs_sumsq_total[cluster2][val]

            cs_sumsq_total_new.append(sumsq_new)

            centroid_list_new = []
            standard_deviation_list_new= []

            for val in range(dimension):
                centroid = sum_new[val] / n_new
                variance = (sumsq_new[val]/ n_new) - (sum_new[val] / n_new) ** 2
                standard_deviation = math.sqrt(variance)
                centroid_list_new.append(centroid)
                standard_deviation_list_new.append(standard_deviation)

            cs_centroid_total_new.append(centroid_list_new)
            cs_standard_deviation_total_new.append(standard_deviation_list_new)

        for i in range(len(cs_total)):
            if i not in indexes_merged:
                cs_total_new.append(cs_total[i])
                cs_sum_total_new.append(cs_sum_total[i])
                cs_sumsq_total_new.append(cs_sumsq_total[i])
                cs_centroid_total_new.append(cs_centroid_total[i])
                cs_standard_deviation_total_new.append(cs_standard_deviation_total[i])

        return cs_total_new,cs_sum_total_new,cs_sumsq_total_new,cs_centroid_total_new,cs_standard_deviation_total_new

    output_list.append(["Round 1", len(ds_cluster_check_list), len(cs_n_list), sum(cs_n_list), len(rs_new)])

    ds_n_list, ds_sum_list, ds_sumsq, rs_list_2,cs_n_list_1,cs_sum_list_1,cs_sumsq_1=load_new_points(data2,ds_centroid,ds_standard_deviation,ds_n_list,ds_sum_list,ds_sumsq,cs_n_list,cs_sum_list,cs_sumsq)
    ds_centroid, ds_standard_deviation = calculate_ds_centroid(ds_n_list, ds_sum_list, ds_sumsq)
    cs_centroid_1, cs_standard_deviation_1 = calculate_cs_centroid(cs_n_list_1, cs_sum_list_1, cs_sumsq_1)

    rs_new_2, cs_points_2 = k_means_on_rs(rs_list_2)
    cs_n_list_2, cs_sum_list_2, cs_sumsq_2 = initialize_cs(cs_points_2, rs_list_2)
    cs_centroid_2, cs_standard_deviation_2 = calculate_cs_centroid(cs_n_list_2, cs_sum_list_2, cs_sumsq_2)

    cs_n_list_3, cs_sum_list_3, cs_sumsq_3,cs_centroid_3,cs_standard_deviation_3=find_merge_cs_cluster(cs_n_list_1,cs_sum_list_1,cs_sumsq_1,cs_centroid_1,cs_standard_deviation_1,cs_n_list_2,cs_sum_list_2,cs_sumsq_2,cs_centroid_2,cs_standard_deviation_2)
    output_list.append(["Round 2", len(ds_cluster_check_list), len(cs_n_list_3), sum(cs_n_list_3), len(rs_new_2)])



    ds_n_list, ds_sum_list, ds_sumsq, rs_list_3,cs_n_list_3, cs_sum_list_3, cs_sumsq_3 = load_new_points(data3, ds_centroid, ds_standard_deviation, ds_n_list,ds_sum_list, ds_sumsq,cs_n_list_3, cs_sum_list_3, cs_sumsq_3)
    ds_centroid, ds_standard_deviation = calculate_ds_centroid(ds_n_list, ds_sum_list, ds_sumsq)


    rs_new_3, cs_points_4 = k_means_on_rs(rs_list_3)
    cs_n_list_4, cs_sum_list_4, cs_sumsq_4 = initialize_cs(cs_points_4, rs_list_3)
    cs_centroid_4, cs_standard_deviation_4 = calculate_cs_centroid(cs_n_list_4, cs_sum_list_4, cs_sumsq_4)

    cs_n_list_5, cs_sum_list_5, cs_sumsq_5,cs_centroid_5,cs_standard_deviation_5=find_merge_cs_cluster(cs_n_list_3,cs_sum_list_3,cs_sumsq_3,cs_centroid_3,cs_standard_deviation_3,cs_n_list_4,cs_sum_list_4,cs_sumsq_4,cs_centroid_4,cs_standard_deviation_4)
    output_list.append(["Round 3", len(ds_cluster_check_list), len(cs_n_list_5), sum(cs_n_list_5), len(rs_new_3)])



    ds_n_list, ds_sum_list, ds_sumsq, rs_list_4,cs_n_list_5, cs_sum_list_5, cs_sumsq_5= load_new_points(data4, ds_centroid, ds_standard_deviation, ds_n_list,ds_sum_list, ds_sumsq,cs_n_list_5, cs_sum_list_5, cs_sumsq_5)
    ds_centroid, ds_standard_deviation = calculate_ds_centroid(ds_n_list, ds_sum_list, ds_sumsq)

    rs_new_4, cs_points_5 = k_means_on_rs(rs_list_4)
    cs_n_list_6, cs_sum_list_6, cs_sumsq_6 = initialize_cs(cs_points_5, rs_list_4)
    cs_centroid_6, cs_standard_deviation_6 = calculate_cs_centroid(cs_n_list_6, cs_sum_list_6, cs_sumsq_6)
    cs_n_list_7, cs_sum_list_7, cs_sumsq_7,cs_centroid_7,cs_standard_deviation_7=find_merge_cs_cluster(cs_n_list_5,cs_sum_list_5,cs_sumsq_5,cs_centroid_5,cs_standard_deviation_5,cs_n_list_6,cs_sum_list_6,cs_sumsq_6,cs_centroid_6,cs_standard_deviation_6)

    output_list.append(["Round 4", len(ds_cluster_check_list), len(cs_n_list_7), sum(cs_n_list_7), len(rs_new_4)])


    ds_n_list, ds_sum_list, ds_sumsq, rs_list_5,cs_n_list_7, cs_sum_list_7, cs_sumsq_7= load_new_points(data5, ds_centroid, ds_standard_deviation, ds_n_list,ds_sum_list, ds_sumsq,cs_n_list_7, cs_sum_list_7, cs_sumsq_7)
    ds_centroid, ds_standard_deviation = calculate_ds_centroid(ds_n_list, ds_sum_list, ds_sumsq)

    rs_new_5, cs_points_6 = k_means_on_rs(rs_list_5)
    cs_n_list_8, cs_sum_list_8, cs_sumsq_8 = initialize_cs(cs_points_6, rs_list_5)
    cs_centroid_8, cs_standard_deviation_8 = calculate_cs_centroid(cs_n_list_8, cs_sum_list_8, cs_sumsq_8)
    cs_n_list_9, cs_sum_list_9, cs_sumsq_9,cs_centroid_9,cs_standard_deviation_9=find_merge_cs_cluster(cs_n_list_7,cs_sum_list_7,cs_sumsq_7,cs_centroid_7,cs_standard_deviation_7,cs_n_list_8,cs_sum_list_8,cs_sumsq_8,cs_centroid_8,cs_standard_deviation_8)

    output_list.append(["Round 5", len(ds_cluster_check_list), len(cs_n_list_9), sum(cs_n_list_9), len(rs_new_5)])

    def merge_ds_cs_clusters(ds_n_list, ds_sum_list, ds_sumsq, ds_centroid, ds_standard_deviation, cs_n_list_9,cs_sum_list_9, cs_sumsq_9, cs_centroid_9, cs_standard_deviation_9):
            cs_total_new = []
            cs_sum_total_new = []
            cs_sumsq_total_new = []
            cs_centroid_total_new = []
            cs_standard_deviation_total_new = []
            clusters_to_merge = []
            merged = []
            for i in range(len(cs_n_list_9)):
                for j in range(len(ds_n_list)):
                    if i not in merged:
                        distance = mahanobolis_distance(cs_centroid_9[i], ds_centroid[j],
                                                            ds_standard_deviation[j])
                        if distance < 2 * math.sqrt(dimension):
                            clusters_to_merge.append([i, j])
                            merged.append(i)


            for i in clusters_to_merge:
                cluster_cs = i[0]
                cluster_ds = i[1]
                ds_n_list[cluster_ds]+=cs_n_list_9[cluster_cs]

                for val in range(dimension):
                    ds_sum_list[cluster_ds][val]+= cs_sum_list_9[cluster_cs][val]


                for val in range(dimension):
                    ds_sumsq[cluster_ds][val] += cs_sumsq_9[cluster_cs][val]

            ds_centroid, ds_standard_deviation = calculate_ds_centroid(ds_n_list, ds_sum_list, ds_sumsq)

            for i in range(len(cs_n_list_9)):
                if i not in merged:
                    cs_total_new.append(cs_n_list_9[i])
                    cs_sum_total_new.append(cs_sum_list_9[i])
                    cs_sumsq_total_new.append(cs_sumsq_9[i])
                    cs_centroid_total_new.append(cs_centroid_9[i])
                    cs_standard_deviation_total_new.append(cs_standard_deviation_9[i])



            return ds_n_list, ds_sum_list, ds_sumsq, ds_centroid, ds_standard_deviation,cs_total_new, cs_sum_total_new, cs_sumsq_total_new, cs_centroid_total_new, cs_standard_deviation_total_new


    ds_n_list, ds_sum_list, ds_sumsq, ds_centroid, ds_standard_deviation, cs_total_new, cs_sum_total_new, cs_sumsq_total_new, cs_centroid_total_new, cs_standard_deviation_total_new=merge_ds_cs_clusters(ds_n_list, ds_sum_list, ds_sumsq, ds_centroid, ds_standard_deviation, cs_n_list_9,cs_sum_list_9, cs_sumsq_9, cs_centroid_9, cs_standard_deviation_9)

    data_label=data.map(lambda row:((str(row[2])),int(row[0])))
    data_label1=spark_context.parallelize(ds_cluster_check_list).map(lambda row:((row[0]),row[1]))
    data1=data_label.join(data_label1)

    data_label = data.map(lambda row: ((str([row[i] for i in range(2, len(row))])), int(row[0])))
    data_label1 = spark_context.parallelize(ds_cluster_check_list).map(lambda row: ((row[0]), row[1]))
    data_index = data_label.join(data_label1).map(lambda row: ((row[1][1]), [row[1][0]])).reduceByKey(operator.add).collectAsMap()
    data_dict = dict()
    for key, value in data_index.items():
        for i in value:
            data_dict[i] = key

    data_dict = dict(sorted(data_dict.items()))

    data_label2=spark_context.parallelize(cs_cluster_check_list).map(lambda row:((row[0]),row[1]))
    data_index2 = data_label.join(data_label2).map(lambda row: ((row[1][1]), [row[1][0]])).reduceByKey(operator.add).collectAsMap()
    data_dict2 = dict()
    for key, value in data_index2.items():
        for i in value:
            data_dict2[i] = key

    data_dict2 = dict(sorted(data_dict2.items()))


    def write(data_dict, output_list,data_dict2):
        file.write("The intermediate results:" + "\n")
        for i in output_list:
            file.write(",".join(list(map(str, i))) + '\n')
        file.write("\n")
        file.write("The clustering results:" + "\n")
        for key, value in data_dict.items():
            file.write(str(key) + "," + str(value) + '\n')
        for key, value in data_dict2.items():
            file.write(str(key) + "," + str(value) + '\n')



    file = open(sys.argv[3], 'w')
    write(data_dict, output_list,data_dict2)
    file.close()
