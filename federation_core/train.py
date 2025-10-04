import logging
import torch
from torch import nn
from device import device  # 你的 device 定义（例如 torch.device("cuda") / cpu）
# 移除全局logger定义，使用handle传入的logger

def _get_data_iterator(dataset):
    """
    兼容原来 dataset 结构：如果 dataset 是 (something, iterator)，返回 iterator，
    否则假设 dataset 本身是 iterable (DataLoader) 并返回它。
    """
    try:
        # 如果 dataset 是 tuple(pair)，且第二项是 data iterator
        if isinstance(dataset, (list, tuple)) and len(dataset) >= 2:
            return dataset[1]
    except Exception:
        pass
    return dataset


def standard_train(epoch, dataset_size, client, params, model, dataset, handle):
    """
    优化后的训练函数（GPU 优先、AMP 支持、GPU 上批量毒化、延迟同步统计）
    参数与原来基本保持一致。
    """
    # 使用当前任务的logger
    logger = handle.logger
    
    model.train()
    index = -1
    local_epoch = int(params.get('local_epochs', 1))
    lr = float(params.get('lr', 0.01))

    # 将模型放到目标设备
    model.to(device)

    # 损失与优化器
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=float(params.get('momentum', 0.9)),
        weight_decay=float(params.get('decay', 0.0))
    )

    data_iterator = _get_data_iterator(dataset)

    # 若 dataset 是 DataLoader，确保用户设置 pin_memory=True 和合理 num_workers，否则这一部分可能仍然是瓶颈。
    use_amp = (device.type == 'cuda') and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 训练循环
    for local_ep in range(local_epoch):
        # 在 GPU 上延迟统计（tensor）
        total_loss_tensor = torch.tensor(0.0, device=device)
        correct_tensor = torch.tensor(0, device=device, dtype=torch.long)
        processed = 0

        # iterate
        for batch_idx, batch in enumerate(data_iterator):
            # 兼容 dataloader 返回 (images, labels) 或 batch 本身就是 (images, labels)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, labels = batch[0], batch[1]
            else:
                # 无法解析 batch，跳过
                continue

            # 防护：若 batch 空则跳过
            if images is None or labels is None:
                continue

            # 移动到设备（non_blocking 以配合 pin_memory）
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.size(0)
            if batch_size == 0:
                continue

            # 前向/反向/优化 使用 AMP（若可用）
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = loss_func(outputs, labels)
                # backward & step via scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

            # 延迟统计：在 GPU 上累加（避免 .item() 导致同步）
            total_loss_tensor += loss.detach() * batch_size
            preds = outputs.argmax(dim=1)
            correct_tensor += preds.eq(labels).sum()
            processed += batch_size

        # epoch 结束，移动统计到 cpu 并记录
        if processed == 0:
            avg_loss = float('nan')
            accuracy = float('nan')
        else:
            # 将 tensor 移到 cpu 并转换为 python float/int
            avg_loss = (total_loss_tensor / processed).cpu().item()
            accuracy = (100.0 * correct_tensor.float() / processed).cpu().item()

        logger.info('___Local_Train , g_epoch {:3d}, l_epoch {:3d}, local model {},  Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
            epoch, local_ep + 1, client, avg_loss, int(correct_tensor.cpu().item()), processed, accuracy))

    # 返回模型参数副本（与原来接口一致）
    return model.state_dict()