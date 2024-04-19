import os
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

print(torch.__version__)
print(torch.distributed.is_available())
print(rpc.is_available())


def run(rank, world_size):
    print("torch.distributed.is_initialized()", torch.distributed.is_initialized())

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    rpc.init_rpc(
        "worker_"+str(rank),
        world_size=world_size,
        rank=rank)

    print("torch.distributed.is_initialized()", torch.distributed.is_initialized())

    rpc.shutdown()


def main():
    a = mp.Process(target=run, args=(0,2))
    b = mp.Process(target=run, args=(1,2))

    a.start()
    b.start()

    a.join()
    b.join()


if __name__ == "__main__":
    main()
