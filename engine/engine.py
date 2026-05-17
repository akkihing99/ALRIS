import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from loguru import logger
from utils.dataset import tokenize
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather,
                        trainMetricGPU)


def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, iou_meter, pr_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    # size_list = [320, 352, 384, 416, 448, 480, 512]
    # idx = np.random.choice(len(size_list))
    # new_size = size_list[idx]

    for i, (image, text, target, *_) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # data
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).unsqueeze(1)

        # # multi-scale training
        # image = F.interpolate(image, size=(new_size, new_size), mode='bilinear')

        # forward
        with amp.autocast():
            pred, target, loss = model(image, text, target)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        # metric
        iou, pr5 = trainMetricGPU(pred, target, 0.35, 0.5)
        dist.all_reduce(loss.detach())
        dist.all_reduce(iou)
        dist.all_reduce(pr5)
        loss = loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr5 = pr5 / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)



@torch.no_grad()
def validate(val_loader, model, epoch, args):
    iou_list = []
    total_inter, total_union = 0.0, 0.0
    model.eval()
    time.sleep(2)
    for imgs, texts, param in val_loader:
        imgs = imgs.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)
        preds = model(imgs, texts)
        preds = torch.sigmoid(preds)
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds, size=imgs.shape[-2:], mode='bicubic', align_corners=True).squeeze(1)
        for pred, mask_dir, mat, ori_size in zip(preds, param['mask_dir'], param['inverse'], param['ori_size']):
            h, w = np.array(ori_size)
            mat = np.array(mat)
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h), flags=cv2.INTER_CUBIC, borderValue=0.)
            pred = np.array(pred > 0.35)
            mask = cv2.imread(mask_dir, flags=cv2.IMREAD_GRAYSCALE) / 255.
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            total_inter += np.sum(inter)
            total_union += np.sum(union)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)

    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(imgs.device)
    iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    miou = iou_list.mean() * 100.
    oiou = total_inter / (total_union + 1e-6) * 100.
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  mIoU={:.2f} oIoU={:.2f}'.format(
        epoch, args.epochs, miou.item(), oiou.item())
    logger.info(head + temp)
    return miou.item(), oiou.item(), prec


@torch.no_grad()
def inference(test_loader, model, args, test_split='val'):
    iou_list = []
    total_inter, total_union = 0.0, 0.0
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    model.eval()
    time.sleep(2)
    for img, param in tbar:
        img = img.cuda(non_blocking=True)
        mask = cv2.imread(param['mask_dir'][0], flags=cv2.IMREAD_GRAYSCALE)
        if args.visualize:
            seg_id = param['seg_id'][0].cpu().numpy()
            img_name = '{}-img.jpg'.format(seg_id)
            mask_name = '{}-mask.png'.format(seg_id)
            cv2.imwrite(filename=os.path.join(args.vis_dir, img_name), img=param['ori_img'][0].cpu().numpy())
            cv2.imwrite(filename=os.path.join(args.vis_dir, mask_name), img=mask)

        for sent in param['sents']:
            mask = mask / 255.
            text = tokenize(sent, args.word_len, True).cuda(non_blocking=True)
            pred = model(img, text)
            pred = torch.sigmoid(pred)
            if pred.shape[-2:] != img.shape[-2:]:
                pred = F.interpolate(pred, size=img.shape[-2:], mode='bicubic', align_corners=True).squeeze()
            h, w = param['ori_size'].numpy()[0]
            mat = param['inverse'].numpy()[0]
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h), flags=cv2.INTER_CUBIC, borderValue=0.)
            pred = np.array(pred > 0.35)

            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            total_inter += np.sum(inter)
            total_union += np.sum(union)

            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
            if args.visualize:
                pred = np.array(pred * 255, dtype=np.uint8)
                sent = "_".join(sent[0].split(" "))
                pred_name = '{}-iou={:.2f}-{}.png'.format(seg_id, iou*100, sent)
                cv2.imwrite(filename=os.path.join(args.vis_dir, pred_name), img=pred)

    logger.info('=> Metric Calculation <=')
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(img.device)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    oIoU = total_inter / (total_union + 1e-6)

    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
    logger.info('[{}] mIoU={:.2f}  oIoU={:.2f}'.format(test_split, 100.*iou.item(), 100.*oIoU))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100.*v))

    return iou.item(), oIoU, prec


@torch.no_grad()
def inference_ddp(test_loader, model, args):
    iou_list = []
    total_inter, total_union = 0.0, 0.0
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tbar = tqdm(test_loader, desc='Inference:', ncols=100) if rank == 0 else test_loader
    model.eval()
    time.sleep(2)
    
    for img, param in tbar:
        img = img.cuda(non_blocking=True)
        mask = cv2.imread(param['mask_dir'][0], flags=cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask > 127).astype(np.bool_)

        if args.visualize and rank == 0:
            seg_id = param['seg_id'][0].cpu().numpy()
            img_name = '{}-img.jpg'.format(seg_id)
            mask_name = '{}-mask.png'.format(seg_id)
            cv2.imwrite(filename=os.path.join(args.vis_dir, img_name), img=param['ori_img'][0].cpu().numpy())
            cv2.imwrite(filename=os.path.join(args.vis_dir, mask_name), img=mask)

        for sent in param['sents']:
            text = tokenize(sent, args.word_len, True).cuda(non_blocking=True)
            pred = model(img, text)
            pred = torch.sigmoid(pred)
            if pred.shape[-2:] != img.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred, size=img.shape[-2:], mode='bicubic', align_corners=True
                ).squeeze()
            h, w = param['ori_size'].numpy()[0]
            mat = param['inverse'].numpy()[0]
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h), flags=cv2.INTER_CUBIC, borderValue=0.)
            pred_bin = (pred > 0.35)

            inter = np.logical_and(pred_bin, mask_bin)
            union = np.logical_or(pred_bin, mask_bin)
            total_inter += np.sum(inter)
            total_union += np.sum(union)

            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)

            if args.visualize and rank == 0:
                pred_vis = np.array(pred_bin * 255, dtype=np.uint8)
                sent_name = "_".join(sent[0].split(" "))
                pred_name = '{}-iou={:.2f}-{}.png'.format(seg_id, iou * 100, sent_name)
                cv2.imwrite(filename=os.path.join(args.vis_dir, pred_name), img=pred_vis)

    iou_tensor = torch.tensor(iou_list, device=img.device, dtype=torch.float64)
    all_ious = concat_all_gather_varlen(iou_tensor)

    # All-reduce inter and union
    inter_tensor = torch.tensor([total_inter], device=img.device)
    union_tensor = torch.tensor([total_union], device=img.device)
    dist.all_reduce(inter_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(union_tensor, op=dist.ReduceOp.SUM)

    total_inter = inter_tensor.item()
    total_union = union_tensor.item()

    if rank == 0:
        prec_list = []
        for thres in torch.arange(0.5, 1.0, 0.1):
            tmp = (all_ious > thres).float().mean()
            prec_list.append(tmp)
        iou = all_ious.mean()
        oIoU = total_inter / (total_union + 1e-6)
        prec = {}
        for i, thres in enumerate(range(5, 10)):
            key = 'Pr@{}'.format(thres * 10)
            value = prec_list[i].item()
            prec[key] = value

        logger.info('=> Metric Calculation <=')
        logger.info('mIoU={:.2f}  oIoU={:.2f}'.format(100. * iou.item(), 100. * oIoU))
        for k, v in prec.items():
            logger.info('{}: {:.2f}.'.format(k, 100. * v))
        return iou.item(), oIoU, prec
    else:
        return None, None, None

def concat_all_gather_varlen(var):
    """All-gather for tensors of variable lengths (1D float tensors)."""
    # 1. Gather lengths
    local_len = torch.tensor([var.numel()], device=var.device)
    all_lens = [torch.zeros_like(local_len) for _ in range(dist.get_world_size())]
    dist.all_gather(all_lens, local_len)
    all_lens = [int(l.item()) for l in all_lens]
    max_len = max(all_lens)

    # 2. Pad to max length
    if var.numel() < max_len:
        pad = torch.zeros(max_len - var.numel(), device=var.device, dtype=var.dtype)
        var = torch.cat([var, pad], dim=0)

    # 3. All-gather
    gathered = [torch.zeros_like(var) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, var)

    # 4. Slice and concat
    valid_vals = [g[:l] for g, l in zip(gathered, all_lens)]
    return torch.cat(valid_vals, dim=0)