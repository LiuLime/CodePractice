import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os


def ddp_setup():


    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def gather_data(data, world_size):
    """
    收集所有进程数据:
        dist.barrier()
            同步屏障,确保同步
        gather_data = [None for _ in range(world_size)]
            初始化参与分布式计算的进程数列表
        dist.all_gather_object(gather_data, data)
            集体通信,每个进程将其data对象发送给其他所有进程,同时接收其他所有进程的data对象。
            gather_data列表将包含来自所有进程的数据。
    """
    dist.barrier()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data


def order_as_origin(indices:list, preds:list) -> tuple[list, list]:
    combine = list(zip(indices, preds))
    combine.sort(key=lambda x: x[0])
    sort_indices, sort_preds = zip(*combine)
    return list(sort_indices), list(sort_preds)


def preds(rank, world_size, test_iter, model):
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    model.eval()
    labels = []
    indices = []  
    with torch.no_grad():
        for idx, input in test_iter:
            input = input.to(rank)
            preds_logit = model(input)

            preds = torch.argmax(preds_logit, dim=-1).to('cpu').detach().numpy()
            labels.extend(preds)
            indices.extend(idx.tolist())  # 将索引添加到列表中
    
    gather_indices = gather_data(indices, world_size) 
    gather_preds = gather_data(labels, world_size)

    if rank == 0:
        flatten_indices = [index for rank_indices in gather_indices for index in rank_indices]
        flatten_labels = [label for rank_labels in gather_preds for label in rank_labels]
        order_indices, order_labels = order_as_origin(flatten_indices, flatten_labels)
        return order_indices, order_labels
    else:
        return None, None
