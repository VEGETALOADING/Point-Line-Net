import math
import sys
import time

import numpy as np
import torch

import train_utils.distributed_utils as utils
from .coco_eval import EvalCOCOMetric

def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)
    mBoxLoss = torch.zeros(1).to(device)
    mKpLoss = torch.zeros(1).to(device)
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            loss_keypoint = loss_dict["loss_keypoint"]
            loss_box = loss_dict["loss_classifier"] \
                       + loss_dict["loss_box_reg"] \
                       + loss_dict["loss_objectness"] \
                       + loss_dict["loss_rpn_box_reg"]
            losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        loss_keypoint_value = loss_keypoint.item()
        loss_box_value = loss_box.item()
        mloss = (mloss * i + loss_value) / (i + 1)
        mKpLoss = (mKpLoss * i + loss_keypoint_value) / (i + 1)
        mBoxLoss = (mBoxLoss * i + loss_box_value) / (i + 1)
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr, mKpLoss, mBoxLoss

def distance(x0, y0, x1, y1, x2, y2):
    k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / \
        ((x2 - x1) ** 2 + (y2 - y1) ** 2) * 1.0

    x_foot = k * (x2 - x1) + x1
    y_foot = k * (y2 - y1) + y1

    pos_on = ((x_foot - x1) > 0) ^ ((x_foot - x2) > 0)
    pos_not_on = ~ pos_on

    dist = torch.zeros_like(x0)

    dist[pos_on] = torch.norm(torch.stack((x_foot[pos_on] - x0[pos_on], y_foot[pos_on] - y0[pos_on]), dim=0),
                              p=2,
                              dim=0,
                              keepdim=False)

    a = torch.norm(torch.stack((x1[pos_not_on] - x0[pos_not_on], y1[pos_not_on] - y0[pos_not_on]), dim=0),
               p=2,
               dim=0,
               keepdim=False)
    b = torch.norm(torch.stack((x2[pos_not_on] - x0[pos_not_on], y2[pos_not_on] - y0[pos_not_on]), dim=0),
               p=2,
               dim=0,
               keepdim=False)
    dist[pos_not_on] = torch.where(a < b, a, b)

    return torch.mean(torch.min(dist.view(10, -1), dim=1).values)
def calculate_distance(predict_keyPoints, gt_keyPoints):

    predict_keyPoints = predict_keyPoints[..., :-1]
    mean_distance_in_image = 0
    for predict_instance, gt_instance in zip(predict_keyPoints, gt_keyPoints):

        gt_instance = gt_instance[(gt_instance[:, 0] != 0) & (gt_instance[:, 1] != 0)]

        gt_instance = torch.cat([
            gt_instance[0].view(1, 2),
            gt_instance[1:-1, :].repeat(1, 2).view(-1, 2),
            gt_instance[-1].view(1, 2)
        ]).view(-1, 4).repeat(10, 1).view(-1, 4)

        predict_instance = predict_instance.repeat(1, int(gt_instance.shape[0] / 10)).view(-1, 2)
        mean_distance_in_instance = distance(predict_instance[:, 0],
                 predict_instance[:, 1],
                 gt_instance[:, 0],
                 gt_instance[:, 1],
                 gt_instance[:, 2],
                 gt_instance[:, 3]).item()
        mean_distance_in_image += mean_distance_in_instance

    mean_distance_in_image /= predict_keyPoints.shape[0]
    return mean_distance_in_image
@torch.no_grad()
def evaluate(model, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "
    det_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="bbox", results_file_name="det_results.json")
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image, targets)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        det_metric.update(targets, outputs)
        metric_logger.update(model_time=model_time)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    det_metric.synchronize_results()

    if utils.is_main_process():
        coco_info = det_metric.evaluate()
    else:
        coco_info = None

    return coco_info

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
def matchPredictToGtbox(predict_boxes, gt_boxes, gt_linepoints):
    if gt_boxes.numel() == 0:
        device = predict_boxes.device
        linepoints = torch.zeros(
            (predict_boxes.shape), dtype=torch.int64, device=device
        )
    else:
        match_quality_matrix = box_iou(gt_boxes, predict_boxes)
        matched_vals, matches = match_quality_matrix.max(dim=0)
        linepoints = gt_linepoints[matches]

        return linepoints