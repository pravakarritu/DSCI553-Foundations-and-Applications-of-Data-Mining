import sys
import time
import random
from blackbox import BlackBox

if __name__ == "__main__":
    random.seed(553)
    data = sys.argv[1]
    file = open(sys.argv[4], 'w')
    file.write('seqnum,0_id,20_id,40_id,60_id,80_id' + '\n')
    bx = BlackBox()
    num_of_asks = int(sys.argv[3])
    stream_size = int(sys.argv[2])
    results = []
    reservoir = []


    def write(values):
        for val in values:
            file.write(",".join(list(map(str, val))) + "\n")


    counter = 1
    for i in range(num_of_asks):
        results = []
        
        stream_users = bx.ask(data, stream_size)
        if i == 0:
            reservoir = stream_users
            results.append([(i + 1) * 100, reservoir[0], reservoir[20], reservoir[40], reservoir[60], reservoir[80]])
            counter=100
        else:

            for j in range(len(stream_users)):
                counter += 1
                probability = 100 / counter
                probability_random = random.random()
                if (probability_random < probability) and (stream_users[j] not in reservoir):
                    index = random.randint(0, 99)
                    reservoir[index] = stream_users[j]
                
            results.append([(i + 1) * 100, reservoir[0], reservoir[20], reservoir[40], reservoir[60], reservoir[80]])
        write(results)

    file.close()
    