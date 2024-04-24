import os
import sys

import torch
import random
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

print(torch.__version__)
print(torch.distributed.is_available())
print(rpc.is_available())


def initialize(rank, world_size):
    print("torch.distributed.is_initialized()", torch.distributed.is_initialized())

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    rpc.init_rpc(
        "worker_" + str(rank),
        world_size=world_size,
        rank=rank)

    print("torch.distributed.is_initialized()", torch.distributed.is_initialized())

    if rank != 0:
        rpc.shutdown()


def conditional_rerun(batch, batch_size, rank):
    print("Current batch length is %d on worker %d" % (len(batch), rank))

    if len(batch) < batch_size:
        print("rerunning", rank)
        rpc.rpc_async(rank, producer_function, args=(rank,))

    return


def consumer_function(rank, world_size):
    if rank != 0:
        exit("ERROR: Consumer rank must be 0")

    initialize(rank, world_size)

    batch_size = 16
    batch = list()

    print(world_size)

    futures = dict()

    for i in range(1, world_size):
        futures[i] = rpc.rpc_async(i, producer_function, args=(i,))

    done = False
    while not done:
        for i, future in futures.items():
            if not future.done():
                continue

            else:
                batch.append(future.value())

                if len(batch) < batch_size:
                    futures[i] = rpc.rpc_async(i, producer_function, args=(i,))
                else:

                    done = True
                    break

    print(batch)

    rpc.shutdown()


def producer_function(i):
    print("launching", i)
    while True:
        n = int(random.random() * 100)

        if n > 50:
            print("yippee ", n, " ", i)
            return (n, i)


def main():
    n_processes = 4
    processes = list()

    for i in range(n_processes):
        if i == 0:
            processes.append(mp.Process(target=consumer_function, args=(i, n_processes)))
        else:
            processes.append(mp.Process(target=initialize, args=(i, n_processes)))

        processes[-1].start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
