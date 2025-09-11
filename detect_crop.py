# yolov5/detect_and_crop_v5.py
import json
from pathlib import Path
import os
import torch
import argparse
import cv2
import numpy as np

# === YOLOv5 internals ===
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (non_max_suppression, scale_boxes, check_img_size, LOGGER)
from utils.torch_utils import select_device

# ---------- Config defaults ----------
PAD_FRAC   = 0.12      # padding around bbox
CONF_THR   = 0.25      # confidence threshold
IOU_THR    = 0.50      # IoU threshold
TARGET_H   = 640       # fix crop height, keep aspect
TOP_K      = None      # None = keep all vials
# --------------------------------------

def expand_and_clamp(x1,y1,x2,y2,W,H,pad_frac):
    w, h = x2-x1, y2-y1
    cx, cy = x1+w/2, y1+h/2
    w2, h2 = w*(1+pad_frac), h*(1+pad_frac)
    x1n = max(int(round(cx - w2/2)), 0)
    y1n = max(int(round(cy - h2/2)), 0)
    x2n = min(int(round(cx + w2/2)), W-1)
    y2n = min(int(round(cy + h2/2)), H-1)
    return x1n, y1n, x2n, y2n

def resize_keep_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img, 1.0
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    out = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return out, scale

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, required=True, help='path to vial detector weights')
    p.add_argument('--source',  type=str, required=True, help='image or folder')
    p.add_argument('--imgsz',   nargs='+', type=int, default=[640], help='inference size h,w')
    p.add_argument('--conf',    type=float, default=CONF_THR)
    p.add_argument('--iou',     type=float, default=IOU_THR)
    p.add_argument('--device',  type=str, default='')
    p.add_argument('--outdir',  type=str, default='runs/crops')
    p.add_argument('--pad',     type=float, default=PAD_FRAC)
    p.add_argument('--target_h',type=int, default=TARGET_H)
    p.add_argument('--topk',    type=int, default=-1, help='keep top-K vials by confidence; -1 = all')
    return p.parse_args()

def main():
    args = parse_args()
    out_root = Path(args.outdir)
    out_imgs = out_root / 'images'
    out_root.mkdir(parents=True, exist_ok=True)
    out_imgs.mkdir(parents=True, exist_ok=True)
    manifest = open(out_root / 'manifest.jsonl', 'w')

    # --- Load model ---
    device = select_device(args.device)
    model  = DetectMultiBackend(args.weights, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = args.imgsz * 2 if len(args.imgsz) == 1 else args.imgsz
    imgsz = check_img_size(imgsz, s=stride)

    # --- Dataloader ---
    dataset = LoadImages(args.source, img_size=imgsz, stride=stride, auto=pt)

    for path, im, im0s, vid_cap, s in dataset:
        # to tensor
        im_tensor = torch.from_numpy(im).to(model.device)
        im_tensor = im_tensor.float() / 255.0
        if im_tensor.ndim == 3:
            im_tensor = im_tensor[None]

        # inference
        pred = model(im_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, args.conf, args.iou, classes=None, agnostic=False, max_det=1000)

        im0 = im0s.copy()
        H, W = im0.shape[:2]

        for det_i, det in enumerate(pred):
            if not len(det):
                LOGGER.info(f'{path}: no vial detections')
                continue

            # rescale to original image space
            det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()

            # sort by confidence
            det = det[det[:,4].argsort(descending=True)]
            if args.topk and args.topk > 0:
                det = det[:args.topk]

            for j, (*xyxy, conf, cls) in enumerate(det):
                x1,y1,x2,y2 = map(int, xyxy)
                x1e,y1e,x2e,y2e = expand_and_clamp(x1,y1,x2,y2, W,H, args.pad)

                crop = im0[y1e:y2e, x1e:x2e].copy()
                crop_resized, scale = resize_keep_height(crop, args.target_h)

                stem = Path(path).stem
                out_name = f'{stem}_vial{j:02d}.png'
                out_path = out_imgs / out_name
                cv2.imwrite(str(out_path), crop_resized)

                rec = {
                    'source_image': str(path),
                    'crop_image': str(out_path),
                    'vial_index': int(j),
                    'det_conf': float(conf),
                    'bbox_xyxy_src': [x1, y1, x2, y2],
                    'bbox_xyxy_expanded_src': [x1e, y1e, x2e, y2e],
                    'resize': {'target_h': int(args.target_h), 'scale': float(scale)},
                    'src_size': [int(W), int(H)],
                    'crop_size': [int(crop_resized.shape[1]), int(crop_resized.shape[0])]
                }
                manifest.write(json.dumps(rec) + '\n')

    manifest.close()
    print(f'Crops saved to: {out_imgs}\nManifest: {out_root / "manifest.jsonl"}')

if __name__ == '__main__':
    main()
