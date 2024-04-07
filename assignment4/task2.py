import itertools
import operator
from graphframes import *
import sys
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import os
import itertools
from functools import reduce
import time

if __name__ == "__main__":
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

    users_common_business_rdd1 = users_common_business_rdd.collect()
    users_common_business_rdd2 = users_common_business_rdd.map(lambda row: (row[1], row[0])).collect()
    edges = users_common_business_rdd1 + users_common_business_rdd2

    edges = spark_context.parallelize(edges).map(lambda row: (users_indexed[row[0]], users_indexed[row[1]]))
    nodes = nodes.map(lambda row: users_indexed[list(row)[0]])


    def final_communities(nodes):
        global edges
        global adjacency_list
        global sum_values
        global adjacency_list_degree
        number_of_communities = find_communitites(nodes)
        num_of_communities = len(number_of_communities)
        modularity = -9999
        edge_present = dict()
        for i in number_of_communities:
            for j in i:
                for k in i:
                    edge_present[(j, k)] = 1 if (j, k) in edges else 0

        mod_value = 0
        for i in number_of_communities:
            for j in i:
                for k in i:
                    mod_value += (
                            edge_present[j, k] - float(adjacency_list_degree[j] * adjacency_list_degree[k]) / (
                        float(sum_values)))

        modularity = mod_value / sum_values

        max_community = number_of_communities
        list_communities = []
        edge_present = dict()
        while (len(edges)) > 0:
            change = 0
            while change == 0:
                betweeness_calculation_highest = nodes.map(lambda row: betweeness_value(row)).flatMap(
                    lambda row: [val for val in row]).reduceByKey(operator.add).sortBy(
                    lambda row: (-1 * row[1], row[0]))
                highest_betweeness = betweeness_calculation_highest.collect()[0][1]
                edges_with_highest_betweeness = betweeness_calculation_highest.filter(
                    lambda row: row[1] == highest_betweeness).collect()
                edges = spark_context.parallelize(edges)
                edge_to_remove_list = []
                for i in edges_with_highest_betweeness:
                    edge1 = i[0]
                    edge2 = (i[0][1], i[0][0])
                    edge_to_remove_list.append(edge1)
                    edge_to_remove_list.append(edge2)
                edges = edges.filter(lambda row: row not in edge_to_remove_list)
                adjacency_list = edges.map(lambda row: (row[0], [row[1]])).reduceByKey(operator.add).map(
                    lambda row: (row[0], set(row[1]))).collectAsMap()

                list_communities = find_communitites(nodes)
                if len(list_communities) > num_of_communities:
                    change = 1
                    num_of_communities = len(list_communities)

            for i in list_communities:
                for j in i:
                    for k in i:
                        edge_present[(j, k)] = 1 if (j, k) in edges else 0

            mod_value = 0
            for i in list_communities:
                for j in i:
                    for k in i:
                        mod_value += (
                                edge_present[j, k] - float(adjacency_list_degree[j] * adjacency_list_degree[k]) / (
                            float(sum_values)))

            modularity_value = mod_value / sum_values

            if modularity_value > modularity:
                modularity = modularity_value
                max_community = list_communities

        return max_community


    def find_communitites(nodes):
        global edges
        global adjacency_list
        edges = edges.collect()
        nodes = sorted(nodes.collect())
        list_communities = []
        visited = dict()
        for node in nodes:
            if node not in visited.keys():
                parent_dict = dict()
                levels = dict()
                visited[node] = 1
                levels[0] = [node]
                current_level = 1
                queue = [node, 9999]
                while len(queue):
                    check_node = queue.pop(0)
                    if check_node == 9999:
                        current_level += 1
                        if len(queue) == 0:
                            break
                        queue.append(9999)
                    else:
                        if check_node in adjacency_list.keys():
                            for i in adjacency_list[check_node]:

                                if i not in visited.keys():
                                    queue.append(i)
                                    visited[i] = 1
                                    if current_level in levels.keys():
                                        levels[current_level] += [i]
                                        parent_dict[i] = check_node

                                    else:
                                        levels[current_level] = [i]
                                        parent_dict[i] = check_node
                nodes_in_community = list(reduce(operator.concat, list(levels.values())))
                list_communities.append(nodes_in_community)
        return list_communities


    def betweeness_value(node):
        global edges
        global adjacency_list
        root_node = node
        parent_dict = dict()
        levels = dict()
        visited = dict()
        visited[node] = 1
        levels[0] = [node]
        current_level = 1
        queue = [node, 9999]
        while len(queue):
            check_node = queue.pop(0)
            if check_node == 9999:
                current_level += 1
                if len(queue) == 0:
                    break
                queue.append(9999)
            else:
                if check_node in adjacency_list.keys():
                    for i in adjacency_list[check_node]:

                        if i not in visited.keys():
                            queue.append(i)
                            visited[i] = 1
                            if current_level in levels.keys():
                                levels[current_level] += [i]
                                parent_dict[i] = check_node

                            else:
                                levels[current_level] = [i]
                                parent_dict[i] = check_node
        new_dict = dict()
        current_level = 0
        keys = sorted(levels.keys())
        levels = {key: levels[key] for key in keys}
        for key, value in levels.items():
            new_dict[current_level] = value
            current_level += 1
        parent = dict()
        child = dict()
        root = new_dict.keys()
        edges_new = []
        edges_credit = dict()
        for level in root:
            for node in new_dict[level]:
                if node in adjacency_list.keys():
                    adjacency_list_of_node = adjacency_list[node]
                    next_level = level + 1
                    if next_level in new_dict.keys():
                        nodes_at_next_level = new_dict[next_level]
                        common_edges = set(adjacency_list_of_node) & set(nodes_at_next_level)
                        for i in common_edges:
                            edges_new.append(sorted([node, i]))
                            if node in child.keys():
                                child[node] += [i]
                            else:
                                child[node] = [i]
                            if i in parent.keys():
                                parent[i] += [node]
                            else:
                                parent[i] = [node]
        number_of_paths_from_root = dict()

        for key, value in new_dict.items():
            if key == 0:
                number_of_paths_from_root[value[0]] = 1
            else:
                for i in value:
                    sum_value = 0
                    for j in parent[i]:
                        sum_value += number_of_paths_from_root[j]
                    number_of_paths_from_root[i] = sum_value

        nodes_in_graph = list(reduce(operator.concat, list(new_dict.values())))
        credits_value = dict()
        for i in nodes_in_graph:
            if i == root_node:
                credits_value[i] = 0
            else:
                credits_value[i] = 1
        sorted_reverse = sorted(new_dict.keys(), reverse=True)
        for i in sorted_reverse:
            nodes_values = new_dict[i]
            for j in nodes_values:
                if j != root_node:
                    for k in parent[j]:
                        credits_value[k] += float(credits_value[j]) * (
                                    number_of_paths_from_root[k] / number_of_paths_from_root[j])
                        edges_credit[tuple(sorted((j, k)))] = float(credits_value[j]) * (
                                    number_of_paths_from_root[k] / number_of_paths_from_root[j])

        return [(key, value) for key, value in edges_credit.items()]


    adjacency_list = edges.map(lambda row: (row[0], [row[1]])).reduceByKey(operator.add).map(
        lambda row: (row[0], set(row[1]))).collectAsMap()
    adjacency_list_degree = edges.map(lambda row: (row[0], [row[1]])).reduceByKey(operator.add).map(
        lambda row: (row[0], len(row[1]))).collectAsMap()
    sum_values = sum(adjacency_list_degree.values())
    betweeness_calculation = nodes.map(lambda row: betweeness_value(row)).flatMap(
        lambda row: [val for val in row]).reduceByKey(
        operator.add).map(lambda row: (
        tuple(sorted(tuple((reverse_index_users[row[0][0]], reverse_index_users[row[0][1]])))),
        round(float(row[1] / 2), 5)))
    value = betweeness_calculation.sortBy(lambda row: (-1 * row[1], row[0])).collect()


    def write(value):
        for i in value:
            file1.write(str(i[0]) + "," + str(i[1]) + '\n')


    file1 = open(sys.argv[3], 'w')
    write(value)
    file1.close()

    communities = list(map(lambda row: sorted(map(lambda row: reverse_index_users[row], row)), final_communities(nodes)))

    file2 = open(sys.argv[4], 'w')

    def write_community_value(value):
        value = sorted(list(map(lambda row: (row, len(row)), value)), key=lambda row: (row[1], row[0]))
        for i in value:
            file2.write(str(i[0]).replace("[", "").replace("]", "") + '\n')

    write_community_value(communities)
    file2.close()
