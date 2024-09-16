import torch
import numpy as np

def iou_object_detection_loss(pred_boxes, target_boxes):
    """
    计算 IoU 损失
    :param pred_boxes: 预测框，形状为 (N, 4)，其中 N 是批量大小，4 表示 (x1, y1, x2, y2)
    :param target_boxes: 目标框，形状为 (N, 4)，其中 N 是批量大小，4 表示 (x1, y1, x2, y2)
    :return: IoU 损失值
    """
    # 计算交集
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    # 计算交集面积
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # 计算并集面积
    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_target = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union = area_pred + area_target - intersection

    # 计算 IoU
    iou = intersection / (union + 1e-7)

    # 计算 IoU 损失
    iou_loss = 1 - iou.mean()

    return iou_loss

def iou_semantic_segmentation(pred_mask, target_mask):
    """
    计算预测掩码与真实标签掩码之间的IoU值。
    :param pred_mask: 预测的掩码数组。
    :param target_mask: 真实标签掩码数组。
    返回:
        float: IoU值。
    """
    intersection = np.sum((pred_mask == 1) & (target_mask == 1))
    union = np.sum((pred_mask == 1) | (target_mask == 1))
    if union == 0:
        return 0
    iou = intersection / union
    return iou