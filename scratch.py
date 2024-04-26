import os
import sys
import math

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


def generate_training_set(n, iw=-1):
    x = list()
    y = list()

    for i in range(n):
        x.append(random.random() * 9)
        y.append(math.sin(x[-1]))

    # print(x)
    # print(y)

    x = torch.Tensor(x)
    y = torch.Tensor(y)

    # print(x.shape)
    # print(y.shape)
    # print(x.shape)
    # print(y.shape)

    return x, y


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear1 = torch.nn.Linear(1, 128)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


def consumer_function(rank, world_size):
    if rank != 0:
        exit("ERROR: Consumer rank must be 0")

    initialize(rank, world_size)
    # initialize here

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    batch_size = 8
    batch = list()

    print(world_size)

    futures = dict()

    for i in range(1, world_size):
        futures[i] = rpc.rpc_async(i, producer_function, args=(i, model))

    done = False
    while not done:
        for i, future in futures.items():
            if not future.done():
                continue

            else:
                batch.append(future.value())

                if len(batch) < batch_size:
                    # pass new model here
                    futures[i] = rpc.rpc_async(i, producer_function, args=(i, model))
                else:

                    done = True
                    break

    y_batch = torch.stack(list(zip(*batch))[0], dim=0)
    y_predict_batch = torch.stack(list(zip(*batch))[1], dim=0)

    print(y_batch)
    print(y_batch[0].shape)
    print(y_batch.shape)
    print(y_predict_batch.shape)

    loss = loss_fn(y_predict_batch, y_batch)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    rpc.shutdown()


def producer_function(rank, model):
    # catch latest model
    # run training
    # spit out win
    #
    x, y = generate_training_set(1, rank)

    y_predict = model.forward(x)

    return y, y_predict


def main():
    n_processes = 2
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
