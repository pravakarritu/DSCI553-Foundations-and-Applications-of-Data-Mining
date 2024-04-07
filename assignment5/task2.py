import sys
import time
import binascii
import random
from blackbox import BlackBox

if __name__ == "__main__":
    data = sys.argv[1]
    num_hash = 69997
    a = random.sample(range(0, 9999), 100)
    b = random.sample(range(0, 9999), 100)
    hash_function_list = []

    def hash_val(a1, b1, num_val):
        def function(x):
            return (a1 * x + b1) % num_val
        return function

    for i in range(len(a)):
        hash_function_list.append(hash_val(a[i], b[i], num_hash))

    def myhashs(s):
        result = []
        for f in hash_function_list:
            result.append(f(s))
        return result

    def myhashs_value(s):
        result = myhashs(s)
        result_val = []
        for i in range(len(hash_function_list)):
            hash_value = int(bin(result[i])[2:])
            number_of_training_zeros = len(str(hash_value)) - len(str(hash_value).rstrip('0'))
            result_val.append([i, number_of_training_zeros])
        return result_val

    list_of_total_hash = []
    bit_array = [0 for i in range(num_hash)]
    bx = BlackBox()

    def calculate_hash(list_users1):
        global bit_array
        hash_array = []
        hash_dict = {}
        hash_dict2 = {}

        for i in list_users1:
            list_of_total_hash_values = myhashs_value(i)
            hash_array.append(list_of_total_hash_values)
            for i in list_of_total_hash_values:
                val1 = i[0]
                val2 = i[1]
                if i[0] in hash_dict.keys():
                    hash_dict[val1].append(val2)
                else:
                    hash_dict[val1] = [val2]

        for key, value in hash_dict.items():
            key1 = key % 1000
            val1 = 2 ** max(value)
            if key1 in hash_dict2.keys():
                hash_dict2[key1].append(val1)
            else:
                hash_dict2[key1] = [val1]

        for key, value in hash_dict2.items():
            hash_dict2[key] = float(sum(value) / len(value))

        averages = sorted(list(hash_dict2.values()))
        if len(averages) % 2 == 0:
            median = (averages[int(len(averages) / 2) + 1] + averages[int(len(averages) / 2)]) / 2
        else:
            median = averages[int(len(averages) / 2)]
        return median

    num_of_asks = int(sys.argv[3])
    stream_size = int(sys.argv[2])
    results = []
    users = []
    val11 = []
    total_sum = 0
    for i in range(num_of_asks):
        stream_users = bx.ask(data, stream_size)
        users = []
        for j in stream_users:
            users.append(int(binascii.hexlify(j.encode('utf8')), 16))
        median = int(calculate_hash(users))
        results.append([i, stream_size, median])



    def write(values):
        for val in values:
            file.write(",".join(list(map(str, val))) + "\n")

    file = open(sys.argv[4], 'w')
    file.write('Time,Ground Truth,Estimation' + '\n')
    write(results)
    file.close()
