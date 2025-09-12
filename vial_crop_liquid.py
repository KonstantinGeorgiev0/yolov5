#!/usr/bin/env python3
import argparse, os, json, subprocess
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- YOLOv5 internals ---
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes, check_img_size, LOGGER, increment_path
from utils.torch_utils import select_device

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

def run_stage_a_detect_and_crop(args):
    out_root = Path(args.outdir)
    # crops/expN
    crops_dir = increment_path(out_root / "crops/exp", mkdir=True)
    manifest_fp = crops_dir / f"manifest_{crops_dir.name}.jsonl"
    mf = open(manifest_fp, "w")

    # Load vial model
    device = select_device(args.device)
    vial_model = DetectMultiBackend(args.vial_weights, device=device, dnn=False, data=None, fp16=args.half)
    stride, names, pt = vial_model.stride, vial_model.names, vial_model.pt
    imgsz = args.imgsz * 2 if len(args.imgsz) == 1 else args.imgsz
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadImages(args.source, img_size=imgsz, stride=stride, auto=pt)
    kept = 0
    for path, im, im0s, vid_cap, s in dataset:
        im_tensor = torch.from_numpy(im).to(vial_model.device)
        im_tensor = im_tensor.half() if vial_model.fp16 else im_tensor.float()
        im_tensor /= 255.0
        if im_tensor.ndim == 3:
            im_tensor = im_tensor[None]

        pred = vial_model(im_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, args.vial_conf, args.vial_iou, classes=None, agnostic=False, max_det=1000)

        im0 = im0s.copy()
        H, W = im0.shape[:2]

        for det in pred:
            if not len(det):
                continue
            det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
            det = det[det[:,4].argsort(descending=True)]
            if args.topk and args.topk > 0:
                det = det[:args.topk]

            for j, (*xyxy, conf, cls) in enumerate(det):
                x1,y1,x2,y2 = map(int, xyxy)
                x1e,y1e,x2e,y2e = expand_and_clamp(x1,y1,x2,y2, W,H, args.pad)
                crop = im0[y1e:y2e, x1e:x2e].copy()
                crop_resized, scale = resize_keep_height(crop, args.crop_h)

                stem = Path(path).stem
                out_name = f"{stem}_vial{j:02d}.png"
                out_path = crops_dir / out_name
                cv2.imwrite(str(out_path), crop_resized)

                rec = {
                    "source_image": str(path),
                    "crop_image": str(out_path),
                    "vial_index": int(j),
                    "det_conf": float(conf),
                    "bbox_xyxy_src": [x1, y1, x2, y2],
                    "bbox_xyxy_expanded_src": [x1e, y1e, x2e, y2e],
                    "resize": {"target_h": int(args.crop_h), "scale": float(scale)},
                    "src_size": [int(W), int(H)],
                    "crop_size": [int(crop_resized.shape[1]), int(crop_resized.shape[0])]
                }
                mf.write(json.dumps(rec) + "\n")
                kept += 1

    mf.close()
    LOGGER.info(f"[Stage A] Saved {kept} crops to {crops_dir}")
    return crops_dir, manifest_fp

def run_stage_b_liquid(args, crops_dir):
    """Call YOLOv5 scripts for the liquid model over the crops directory."""
    crops_dir = str(crops_dir)
    out_dir = str(Path(args.outdir) / "liquid_runs")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if args.liquid_task == "detect":
        cmd = [
            "python3", "yolov5/detect.py",
            "--weights", args.liquid_weights,
            "--source", crops_dir,
            "--imgsz", str(args.liquid_imgsz),
            "--conf-thres", str(args.liquid_conf),
            "--iou-thres", str(args.liquid_iou),
            "--save-txt", "--save-conf",
            "--project", out_dir,
            "--line-thickness", "1",
            # "--name", "exp",
            # "--exist-ok"
        ]
    elif args.liquid_task == "segment":
        # Uses YOLOv5 seg pipeline
        cmd = [
            "python3", "yolov5/segment/predict.py",
            "--weights", args.liquid_weights,
            "--source", crops_dir,
            "--imgsz", str(args.liquid_imgsz),
            "--conf", str(args.liquid_conf),
            "--iou", str(args.liquid_iou),
            "--project", out_dir,
            "--name", "exp",
            "--exist-ok"
        ]
    else:
        raise ValueError("liquid_task must be 'detect' or 'segment'.")

    LOGGER.info(f"[Stage B] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return Path(out_dir) / "exp"

# ---------- Post-processing: decide vial state from detection boxes ----------

# Class mapping
GEL_ID = 0
STABLE_ID = 1
AIR_ID = 2
LIQUID_CLASSES = {STABLE_ID, GEL_ID}

# Heuristics
PHASE_LAYER_MIN_GAP = 0.06   # min vertical gap between layer centers (normalized to crop height)
IOU_MERGE_THR       = 0.50   # merge almost identical boxes
STEP_THR            = 22.0   # vertical intensity step backup trigger
GEL_AREA_FRAC_THR   = 0.35   # gel area fraction to classify as gelled; if gel area >=  of liquid area => gel
CONF_MIN              = 0.20   # ignore boxes below this conf
GEL_DOMINANCE_COUNT_THR = 1    # OR if (#gel boxes - #stable boxes) >= 1 => gel
AIR_GAP_THR = 0.08  # % of crop height

def is_liquid_cls(cid: int) -> bool:
    return cid in (GEL_ID, STABLE_ID)

def yolo_line_to_xyxy(line, W, H):
    # "cls cx cy w h [conf]"
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(parts[0])
    cx, cy, w, h = map(float, parts[1:5])
    x1 = (cx - w/2) * W; y1 = (cy - h/2) * H
    x2 = (cx + w/2) * W; y2 = (cy + h/2) * H
    conf = float(parts[5]) if len(parts) >= 6 else 1.0
    return cls, [x1,y1,x2,y2], conf

def box_area(b):
    x1,y1,x2,y2 = b
    return max(0, x2-x1) * max(0, y2-y1)

def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = box_area(a) + box_area(b) - inter
    return inter / (ua + 1e-9)

def merge_boxes_conf(boxes_conf, iou_thr=IOU_MERGE_THR):
    # boxes_conf: list of (conf, box)
    merged = []
    for conf, bb in sorted(boxes_conf, key=lambda x: x[0], reverse=True):
        keep = True
        for _, mb in merged:
            if iou(bb, mb) > iou_thr:
                keep = False; break
        if keep:
            merged.append((conf, bb))
    return merged

def strongest_vertical_step(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    prof = np.median(gray, axis=1)       # vertical profile
    diffs = np.abs(np.diff(prof))
    return float(diffs.max()) if diffs.size else 0.0

def compute_layers_and_gel_fraction(liquid_boxes, crop_h):
    """Return estimated #layers and gel area fraction among liquid area."""
    if not liquid_boxes:
        return 0, 0.0

    # merge near-duplicate boxes
    merged = merge_boxes_conf([(1.0, b) for b in liquid_boxes])  # ignore conf
    centers = []
    for _, (x1,y1,x2,y2) in merged:
        cy = 0.5*(y1+y2)/crop_h
        centers.append(cy)
    centers.sort()

    # count distinct vertically separated bands
    layers = 1
    for i in range(1, len(centers)):
        if centers[i] - centers[i-1] > PHASE_LAYER_MIN_GAP:
            layers += 1

    return layers, 0.0  # gel fraction computed elsewhere

def decide_state_from_boxes(crop_path, label_path, cls_map=(AIR_ID, STABLE_ID, GEL_ID)):
    crop = cv2.imread(str(crop_path))
    if crop is None:
        return {"vial_state": "unknown", "reason": "crop_not_found"}

    H, W = crop.shape[:2]
    liquid_boxes = []
    gel_boxes = []
    total_liquid_area = 0.0
    gel_area = 0.0

    if not Path(label_path).exists():
        # No detections → either air-only or low-conf missed;
        return {"vial_state": "air_only", "num_layers": 0, "gel_area_frac": 0.0, "step": strongest_vertical_step(crop)}

    with open(label_path) as f:
        for line in f:
            parsed = yolo_line_to_xyxy(line, W, H)
            if not parsed:
                continue
            cls, box, conf = parsed
            if cls in LIQUID_CLASSES:
                liquid_boxes.append(box)
                total_liquid_area += box_area(box)
                if cls == GEL_ID:
                    gel_boxes.append(box)
                    gel_area += box_area(box)

    # layer count
    n_layers, _ = compute_layers_and_gel_fraction(liquid_boxes, H)

    # backup: vertical intensity step
    step = strongest_vertical_step(crop)

    # gel fraction
    gel_frac = (gel_area / (total_liquid_area + 1e-9)) if total_liquid_area > 0 else 0.0

    # Decision tree
    if n_layers >= 2 or step > STEP_THR:
        state = "phase_separated"
    else:
        if gel_frac >= GEL_AREA_FRAC_THR:
            state = "gelled"
        else:
            # if any liquid boxes exist -> stable; else air_only
            state = "stable" if total_liquid_area > 0 else "air_only"

    return {
        "vial_state": state,
        "num_layers": int(n_layers),
        "gel_area_frac": float(gel_frac),
        "step": float(step)
    }

def infer_air_present(liquid_boxes_px, H):
    if not liquid_boxes_px:
        return True  # no liquid boxes => air-only
    top_y = min(b[1] for b in liquid_boxes_px)
    return (top_y / float(H)) >= AIR_GAP_THR

def decide_air_stable_gel_from_labels(crop_path: Path, label_path: Path):
    img = cv2.imread(str(crop_path))
    if img is None:
        return {"vial_state": "air", "reason": "crop_not_found"}

    H, W = img.shape[:2]
    if not label_path.exists():
        return {"vial_state": "air", "reason": "no_label_file"}

    liquid_area = 0.0
    gel_area = 0.0
    n_stable = 0
    n_gel = 0
    liquid_boxes_px = []

    with open(label_path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parsed = yolo_line_to_xyxy(raw, W, H)
            if not parsed:
                continue
            cid, box, conf = parsed
            if conf < CONF_MIN:
                continue

            if is_liquid_cls(cid):
                liquid_boxes_px.append(box)
                a = box_area(box)
                liquid_area += a
                if cid == GEL_ID:
                    gel_area += a
                    n_gel += 1
                elif cid == STABLE_ID:
                    n_stable += 1
            # AIR_ID (2) is ignored for liquid metrics

    if liquid_area <= 0:
        return {"vial_state": "air"}  # only air (or nothing detected)

    gel_frac = gel_area / (liquid_area + 1e-9)
    if gel_frac >= GEL_AREA_FRAC_THR or (n_gel - n_stable) >= GEL_DOMINANCE_COUNT_THR:
        return {"vial_state": "gel", "gel_area_frac": float(gel_frac), "n_gel": n_gel, "n_stable": n_stable}

    return {"vial_state": "stable", "gel_area_frac": float(gel_frac), "n_gel": n_gel, "n_stable": n_stable}

def compute_turbidity_profile_v5(crop_bgr):
    # Resize to (100, 500), HSV->V, mean over width
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    turb = cv2.resize(crop_rgb, (100, 500))
    hsv = cv2.cvtColor(turb, cv2.COLOR_RGB2HSV)
    v = np.mean(hsv[:, :, -1], axis=-1)  # (500,)
    v_norm = (v - v.min()) / max(1e-6, (v.max() - v.min()))
    return v, v_norm

def save_turbidity_plot_for_crop(crop_path, v_norm):
    z = np.linspace(0, 1, len(v_norm))
    out_path = Path(crop_path).with_suffix(".turbidity.png")
    plt.figure(figsize=(3.0, 4.0), dpi=120)
    plt.plot(v_norm, z)
    plt.gca().invert_yaxis()
    plt.xlabel("Turbidity (norm)"); plt.ylabel("Normalized Height")
    plt.title(Path(crop_path).name, fontsize=8)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    return str(out_path)

def attach_simple_state(manifest_path, liquid_out_dir, out_path=None):
    manifest_path = Path(manifest_path)
    liquid_out_dir = Path(liquid_out_dir)
    labels_dir = liquid_out_dir / "labels"
    out_path = Path(out_path) if out_path else manifest_path.parent / "manifest_with_state.jsonl"

    with open(manifest_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            crop_path = Path(rec["crop_image"])
            label_path = labels_dir / f"{crop_path.stem}.txt"

            if not label_path.exists():
                alts = sorted(liquid_out_dir.parent.glob(f"exp*/labels/{crop_path.stem}.txt"))
                if alts:
                    label_path = alts[-1]

            state_info = decide_air_stable_gel_from_labels(crop_path, label_path)

            img = cv2.imread(str(crop_path))
            if img is not None:
                v_raw, v_norm = compute_turbidity_profile_v5(img)
                plot_path = save_turbidity_plot_for_crop(crop_path, v_norm)
                state_info.update({
                    "turbidity_plot": plot_path,
                    "turbidity_mean": float(np.mean(v_norm)),
                    "turbidity_maxstep": float(np.max(np.abs(np.diff(v_norm)))) if len(v_norm) > 1 else 0.0
                })

            rec.update(state_info)
            fout.write(json.dumps(rec) + "\n")
    print(f"Wrote: {out_path}")
    return out_path

def attach_states_to_manifest(manifest_path, labels_dir, out_path=None):
    """Read manifest.jsonl, compute state for each crop, and write an updated JSONL."""
    labels_dir = Path(labels_dir)
    out_path = Path(out_path) if out_path else Path(manifest_path).with_name("manifest_with_state.jsonl")

    with open(manifest_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            crop_path = Path(rec["crop_image"])
            label_path = labels_dir / "labels" / (crop_path.stem + ".txt")
            state_info = decide_state_from_boxes(crop_path, label_path)
            rec.update(state_info)  # adds vial_state, num_layers, gel_area_frac, step
            fout.write(json.dumps(rec) + "\n")

    print(f"Wrote: {out_path}")
    return out_path

def parse_args():
    p = argparse.ArgumentParser("Vial → Crop → Liquid")
    # Inputs
    p.add_argument("--source", type=str, required=True, help="image/dir/glob/video for Stage A")
    p.add_argument("--outdir", type=str, default="runs/vial2liquid", help="output root")

    # Stage A (vial)
    p.add_argument("--vial-weights", type=str, required=True, help="vial detector .pt")
    p.add_argument("--imgsz", nargs="+", type=int, default=[640], help="Stage A inference size h,w")
    p.add_argument("--vial-conf", type=float, default=0.65)
    p.add_argument("--vial-iou", type=float, default=0.45)
    p.add_argument("--pad", type=float, default=0.12, help="padding fraction around vial bbox")
    p.add_argument("--crop-h", type=int, default=640, help="resize vial crops to this height")
    p.add_argument("--topk", type=int, default=-1, help="keep top-K vials per image; -1 = all")

    # Stage B (liquid)
    p.add_argument("--liquid-task", choices=["detect","segment"], default="detect",
                   help="use YOLOv5 detect.py or segment/predict.py")
    p.add_argument("--liquid-weights", type=str, required=True, help="liquid model .pt")
    p.add_argument("--liquid-imgsz", type=int, default=640, help="Stage B --img/--imgsz")
    p.add_argument("--liquid-conf", type=float, default=0.25)
    p.add_argument("--liquid-iou", type=float, default=0.50)

    # misc
    p.add_argument("--device", type=str, default="")
    p.add_argument("--half", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    crops_dir, manifest = run_stage_a_detect_and_crop(args)
    liquid_out = run_stage_b_liquid(args, crops_dir)
    # manifest_with_state = attach_states_to_manifest(
    #     manifest_path=str(manifest),
    #     labels_dir=str(liquid_out),
    #     out_path=None
    # )

    # avoid duplication
    stem = Path(manifest).stem.replace("manifest_", "")
    # classification for 3 states (air, stable, gel)
    manifest_with_state = attach_simple_state(
        manifest_path=str(manifest),
        liquid_out_dir=str(liquid_out),
        out_path=str(Path(manifest).with_name(f"manifest_state_{stem}.jsonl"))
    )

    print("\n=== Pipeline complete ===")
    print(f"Crops:       {crops_dir}")
    print(f"Manifest:    {manifest}")
    print(f"Liquid out:  {liquid_out}")
    print(f"With state:  {manifest_with_state}")

if __name__ == "__main__":
    main()
