import os
import torch
import torch.distributed.rpc as rpc

print(torch.__version__)
print(torch.distributed.is_available())
print(rpc.is_available())


def run():
    print("torch.distributed.is_initialized()", torch.distributed.is_initialized())

    init_method = "file:///tmp/rpc"

    if os.path.exists(init_method):
        os.remove(init_method)

    torch.distributed.init_process_group(
        "gloo",
        init_method=init_method,
        rank=0,
        world_size=4)


def main():
    run()


if __name__ == "__main__":
    main()
