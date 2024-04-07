import sys
import time
import binascii
import random
from blackbox import BlackBox
import time

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

    list_of_total_hash = []
    bit_array = [0 for i in range(num_hash)]
    bx = BlackBox()

    def calculate_bit(list_users1):
        global bit_array
        bit_array = [0 for i in range(num_hash)]
        for i in list_users1:
            list_of_total_hash_values = myhashs(i)
            for j in list_of_total_hash_values:
                bit_array[j] = 1

    def calculate(stream_users):
        global previous_set
        false_positive = 0
        true_negative = 0
        for i in stream_users:
            list_of_total_hash1 = myhashs(i)
            for j in list_of_total_hash1:
                if bit_array[j] == 1 and i not in previous_set:
                    false_positive += 1
                elif bit_array[j] == 0 and i not in previous_set:
                    true_negative += 1
        return false_positive / (false_positive + true_negative)

    num_of_asks = int(sys.argv[3])
    stream_size = int(sys.argv[2])
    results = []
    users = []
    for i in range(num_of_asks):
        if i == 0:
            previous_set = []
            stream_users = bx.ask(data, stream_size)
            for j in stream_users:
                users.append(int(binascii.hexlify(j.encode('utf8')), 16))
            false_positive_rate = calculate(users)
            results.append([i, false_positive_rate])
        else:
            previous_set = users
            calculate_bit(previous_set)
            stream_users = bx.ask(data, stream_size)
            users = []
            for j in stream_users:
                users.append(int(binascii.hexlify(j.encode('utf8')), 16))
            false_positive_rate = calculate(users)
            results.append(
                [i, false_positive_rate])

    def write(values):
        for val in values:
            file.write(",".join(list(map(str, val))) + "\n")

    file = open(sys.argv[4], 'w')
    file.write('Time,FPR' + '\n')
   
    write(results)
    file.close()

