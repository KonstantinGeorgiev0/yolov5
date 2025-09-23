#!/usr/bin/env python3
import argparse, os, json, subprocess, shutil
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# matplotlib.use("Agg")

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

def run_stage_a_detect_and_crop(args, manifest_path=None):
    """
    Modified to accept optional manifest path parameter.
    If not provided, creates temporary manifest that will be deleted.
    """
    out_root = Path(args.outdir)
    crops_dir = increment_path(out_root / "crops/exp", mkdir=True)

    # Create temporary manifest if path not provided
    if manifest_path is None:
        temp_manifest = True
        manifest_fp = crops_dir / f"temp_manifest_{crops_dir.name}.jsonl"
    else:
        temp_manifest = False
        manifest_fp = Path(manifest_path)
        manifest_fp.parent.mkdir(parents=True, exist_ok=True)

    # Store basic detection info temporarily
    crop_records = []

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
            det = det[det[:, 4].argsort(descending=True)]
            if args.topk and args.topk > 0:
                det = det[:args.topk]

            for j, (*xyxy, conf, cls) in enumerate(det):
                x1, y1, x2, y2 = map(int, xyxy)
                x1e, y1e, x2e, y2e = expand_and_clamp(x1, y1, x2, y2, W, H, args.pad)
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
                crop_records.append(rec)
                kept += 1

    LOGGER.info(f"[Stage A] Saved {kept} crops to {crops_dir}")

    # Return records and manifest path for later use
    return crops_dir, crop_records, manifest_fp, temp_manifest

def run_stage_b_liquid(args, crops_dir):
    """Call YOLOv5 scripts for the liquid model over the crops directory."""
    crops_dir = Path(crops_dir)
    exp_name = crops_dir.name
    out_dir = Path(args.outdir) / "liquid_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.liquid_task == "detect":
        cmd = [
            "python3", "yolov5/detect.py",
            "--weights", args.liquid_weights,
            "--source", str(crops_dir),
            "--imgsz", str(args.liquid_imgsz),
            "--conf", str(args.liquid_conf),
            "--iou", str(args.liquid_iou),
            "--save-txt", "--save-conf",
            "--project", str(out_dir),
            "--name", exp_name,
            "--line-thickness", "1",
            "--exist-ok",
        ]
    elif args.liquid_task == "segment":
        cmd = [
            "python3", "yolov5/segment/predict.py",
            "--weights", args.liquid_weights,
            "--source", str(crops_dir),
            "--imgsz", str(args.liquid_imgsz),
            "--conf", str(args.liquid_conf),
            "--iou", str(args.liquid_iou),
            "--project", str(out_dir),
            "--name", exp_name,
            "--exist-ok",
        ]
    else:
        raise ValueError("liquid_task must be 'detect' or 'segment'.")

    LOGGER.info(f"[Stage B] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_dir / exp_name

# ---------- Post-processing: decide vial state from detection boxes ----------

# Class mapping
GEL_ID = 0
STABLE_ID = 1
AIR_ID = 2
CAP_ID = 3
LIQUID_CLASSES = {STABLE_ID, GEL_ID}

# Heuristics
# PHASE_LAYER_MIN_GAP = 0.06   # min vertical gap between layer centers (normalized to crop height)
# IOU_MERGE_THR       = 0.50   # merge almost identical boxes
# STEP_THR            = 22.0   # vertical intensity step backup trigger
# GEL_AREA_FRAC_THR   = 0.35   # gel area fraction to classify as gelled; if gel area >=% of liquid area => gel
# CONF_MIN              = 0.20   # ignore boxes below this conf
# GEL_DOMINANCE_COUNT_THR = 1    # OR if (#gel boxes - #stable boxes) >= 1 => gel
# AIR_GAP_THR = 0.08  # % of crop height

def is_liquid_cls(cid: int) -> bool:
    return cid in (GEL_ID, STABLE_ID)

def yolo_line_to_xyxy(line, W, H):
    """
    Converts a YOLO format line into an (x1, y1, x2, y2) bounding box, confidence score
    and class ID.
    """
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
    """
    Calculates the area of a rectangular box given its coordinates.
    """
    x1,y1,x2,y2 = b
    return max(0, x2-x1) * max(0, y2-y1)

def iou_xyxy(a, b):
    """
    Calculates the Intersection over Union (IoU) for two rectangular bounding boxes
    defined in XYXY format.
    """
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / (ua + 1e-9)

def merge_by_iou_desc_conf(dets, iou_thr=0.5):
    """
    Greedy merge: sort by confidence desc; keep a det only if it doesn't
    overlap (IoU>thr) with any already kept det.
    dets: list of dicts with keys 'box' and 'confidence'.
    """
    if not dets:
        return dets
    dets = sorted(dets, key=lambda d: d['confidence'], reverse=True)
    kept = []
    for d in dets:
        bb = d['box']
        if all(iou_xyxy(bb, k['box']) <= iou_thr for k in kept):
            kept.append(d)
    return kept

# ---- State classification ---- #

def detect_phase_separation_logic(detections, vial_height, vial_width, cap_bottom_y=None,
                                  conf_min=0.30, min_area_frac=0.002,
                                  iou_merge_thr=0.5, gap_thr=0.03, span_thr=0.20):
    """
    Determine whether a vial exhibits phase separation (multiple liquid layers) using detected liquid regions.
    """
    # total vial area
    total_area = float(vial_height) * float(vial_width)

    # pre-filter
    liquid = []
    for d in detections:
        if not is_liquid_cls(d['class_id']):
            continue
        if d['confidence'] < conf_min:
            continue
        if cap_bottom_y is not None and d['box'][1] <= cap_bottom_y:
            continue
        if (d['area'] / max(1e-6, total_area)) < min_area_frac:
            continue
        liquid.append(d)

    # merge duplicates
    liquid = merge_by_iou_desc_conf(liquid, iou_thr=iou_merge_thr)
    if len(liquid) >= 2:
        return True, {
            'decision_by': 'count'
        }

    # sort once by y-position (top to bottom)
    detections.sort(key=lambda x: x['center_y'])

    # gap test
    gaps = []
    for i in range( len(detections) - 1 ):
        current_bottom = detections[i]['box'][3] # take y2 of top box
        next_top       = detections[i+1]['box'][1] # take y1 of next box
        gap = (next_top - current_bottom) / float(vial_height)
        gaps.append(gap)
        if gap >= gap_thr:
            return True, {
                'gaps': gaps,
                'max_gap': float(max((gap for gap in gaps), default=0.0)),
                'decision_by': 'gap'
            }

    # span test
    span = (liquid[-1]['center_y'] - liquid[0]['center_y']) / max(1e-6, float(vial_height))
    if span >= span_thr:
        return True, {
            'span': span,
            'decision_by': 'span'
        }

    return False, {
        'gaps': gaps,
        'max_gap': float(max((gap for gap in gaps), default=0.0)),
        'decision_by': 'no_decision'
    }

def detect_phase_separation_from_turbidity(img, cap_bottom_y=None):
    """
    Detect phase separation from turbidity profile with cap-aware region exclusion.
    """
    try:
        # Compute turbidity with exclusion
        v_raw, v_norm, excluded_info = compute_turbidity_profile_with_exclusion(
            img, cap_bottom_y
        )

        if len(v_norm) < 50:  # Minimum profile length
            return False

        # Calculate gradient only in the analysis region
        gradient = np.abs(np.gradient(v_norm))

        # Conservative threshold
        threshold = max(np.mean(gradient) + 2.5 * np.std(gradient), 0.10)

        # Find significant peaks
        significant_peaks = gradient > threshold
        peak_positions = np.where(significant_peaks)[0]

        if len(peak_positions) == 0:
            return False

        # group peaks if they’re close
        min_separation = int(len(v_norm) * 0.1)
        peak_groups = []
        current_group = [peak_positions[0]]

        for i in range(1, len(peak_positions)):
            if peak_positions[i] - peak_positions[i - 1] < min_separation:
                current_group.append(peak_positions[i])
            else:
                peak_groups.append(current_group)
                current_group = [peak_positions[i]]
        peak_groups.append(current_group)

        # collapse each group to one peak
        merged_peaks = [g[np.argmax(gradient[g])] for g in peak_groups]

        # merged_peaks for classification
        if len(merged_peaks) < 2:
            return False

        return True

    except Exception as e:
        print(f"Turbidity analysis failed: {e}")
        return False

def analyze_phase_separation(detections, vial_height):
    """
    Provide detailed analysis of phase separation characteristics.
    """
    liquid_detections = [d for d in detections if is_liquid_cls(d['class_id'])]

    # Sort by vertical position (top to bottom)
    liquid_detections.sort(key=lambda x: x['center_y'])

    layers = []
    current_layer = [liquid_detections[0]]

    # Group detections into layers based on vertical proximity
    layer_tolerance = vial_height * 0.1  # % of vial height

    for i in range(1, len(liquid_detections)):
        det = liquid_detections[i]
        prev_det = current_layer[-1]

        if abs(det['center_y'] - prev_det['center_y']) < layer_tolerance:
            current_layer.append(det)
        else:
            layers.append(current_layer)
            current_layer = [det]

    if current_layer:
        layers.append(current_layer)

    # Analyze each layer
    layer_analysis = []
    for i, layer in enumerate(layers):
        dominant_class = max(set(d['class_id'] for d in layer),
                             key=lambda x: sum(1 for d in layer if d['class_id'] == x))
        total_area = sum(d['area'] for d in layer)
        avg_y = np.mean([d['center_y'] for d in layer]) / vial_height

        layer_analysis.append({
            'layer_index': i,
            'dominant_class': 'gel' if dominant_class == GEL_ID else 'stable',
            'total_area': total_area,
            'normalized_y_position': avg_y,
            'num_detections': len(layer)
        })

    return {
        'num_layers': len(layers),
        'layers': layer_analysis
    }

# def calculate_classification_confidence(detections):
#     """
#     Calculate confidence in the classification based on detection quality.
#     """
#     if not detections:
#         return 0.0
#
#     # Average detection confidence
#     avg_conf = np.mean([d['confidence'] for d in detections])
#
#     # Number of detections
#     detection_score = min(len(detections) / 5.0, 1.0)  # Normalize to max of 5 detections
#
#     # Combine metrics
#     overall_confidence = (avg_conf * 0.7) + (detection_score * 0.3)
#
#     return float(overall_confidence)

def calculate_liquid_volumes(detections, vial_height, vial_width,
                             vial_volume_ml=1, cap_bottom_y=None):
    """
    Estimate liquid volumes from bounding box detections.
    If cap_bottom_y is given, only include liquid below the cap.

    Args:
        detections: list of detection dicts (with 'class_id', 'box', 'area')
        vial_height: total vial height in pixels
        vial_width: total vial width in pixels
        vial_volume_ml: nominal vial volume (mL)
        cap_bottom_y: if not None, exclude liquid above this y-coordinate
    """
    volumes = {}
    # Adjust effective height if cap detected
    effective_height = vial_height
    if cap_bottom_y is not None:
        effective_height = vial_height - cap_bottom_y

    total_vial_area = effective_height * vial_width
    if total_vial_area <= 0:
        return {"gel": 0.0, "stable": 0.0}

    for det in detections:
        if is_liquid_cls(det['class_id']):
            y1 = det['box'][1]
            # Exclude liquid above cap if cap_bottom_y is set
            if cap_bottom_y is not None and y1 <= cap_bottom_y:
                continue

            # Calculate volume fraction
            area_fraction = det['area'] / total_vial_area
            volume_ml = area_fraction * vial_volume_ml

            class_name = "gel" if det['class_id'] == GEL_ID else "stable"
            volumes[class_name] = volumes.get(class_name, 0.0) + volume_ml

    return volumes

def detect_cap_region(detections, img_height):
    """
    Detect cap region from YOLO detections.
    Returns the bottom Y-coordinate of the cap region (or None if no cap detected).

    Args:
        detections: List of detection dictionaries with 'class_id', 'box', etc.
        img_height: Height of the image

    Returns:
        tuple: (cap_bottom_y, cap_info_dict) or (None, None)
    """
    cap_detections = [d for d in detections if d['class_id'] == CAP_ID]

    if not cap_detections:
        return None, None

    # If multiple caps detected, use the topmost one (lowest y1)
    cap_detections.sort(key=lambda d: d['box'][1])  # Sort by top y coordinate
    primary_cap = cap_detections[0]

    cap_info = {
        'detected': True,
        'box': primary_cap['box'],
        'confidence': primary_cap['confidence'],
        'bottom_y': primary_cap['box'][3],  # y2 coordinate
        'top_y': primary_cap['box'][1],  # y1 coordinate
        'height_fraction': (primary_cap['box'][3] - primary_cap['box'][1]) / img_height,
        'num_cap_detections': len(cap_detections)
    }

    return primary_cap['box'][3], cap_info

def enhanced_decide(crop_path: Path, label_path: Path):
    """
    Enhanced four-state classification with cap detection integration.
    """
    # Usage of turbidity analysis for phase separation
    use_turbidity_decide = False

    img = cv2.imread(str(crop_path))
    if img is None:
        return {"vial_state": "unknown", "reason": "crop_not_found"}

    H, W = img.shape[:2]

    # Handle no detection file
    if not label_path.exists():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)

        if mean_brightness > 150 and brightness_std < 30:
            return {
                "vial_state": "only_air",
                "reason": "no_detections_appears_empty",
                "cap_detected": False,
                "brightness_metrics": {
                    "mean_brightness": float(mean_brightness),
                    "brightness_std": float(brightness_std)
                }
            }
        else:
            return {
                "vial_state": "unknown",
                "reason": "detection_failed",
                "cap_detected": False,
                "brightness_metrics": {
                    "mean_brightness": float(mean_brightness),
                    "brightness_std": float(brightness_std)
                }
            }

    # Parse all detections including caps
    detections = []
    liquid_area = 0.0
    gel_area = 0.0
    n_stable = 0
    n_gel = 0
    liquid_boxes = []
    cap_info = None

    with open(label_path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parsed = yolo_line_to_xyxy(raw, W, H)
            if not parsed:
                continue
            cid, box, conf = parsed
            if conf < 0.20:  # CONF_MIN
                continue

            det_dict = {
                'class_id': cid,
                'box': box,
                'confidence': conf,
                'center_y': (box[1] + box[3]) / 2.0,
                'area': box_area(box)
            }
            detections.append(det_dict)

            if is_liquid_cls(cid):
                liquid_boxes.append(box)
                area = box_area(box)
                liquid_area += area

                if cid == GEL_ID:
                    gel_area += area
                    n_gel += 1
                elif cid == STABLE_ID:
                    n_stable += 1

    # Detect cap region
    cap_bottom_y, cap_info = detect_cap_region(detections, H)

    # Filter liquid detections below cap if cap detected
    if cap_bottom_y is not None:
        # Only consider liquid below the cap
        liquid_detections_filtered = []
        for det in detections:
            if is_liquid_cls(det['class_id']):
                if det['box'][1] > cap_bottom_y:  # Top of box is below cap
                    liquid_detections_filtered.append(det)

        # Recalculate liquid areas excluding those above cap
        if liquid_detections_filtered:
            liquid_area = sum(d['area'] for d in liquid_detections_filtered)
            gel_area = sum(d['area'] for d in liquid_detections_filtered if d['class_id'] == GEL_ID)

    # If no liquid detected, it's only air
    if liquid_area <= 0:
        result = {
            "vial_state": "only_air",
            "reason": "no_liquid_detected",
            "total_detections": len(detections)
        }
        if cap_info:
            result["cap_info"] = cap_info
        return result

    # Check for phase separation
    is_phase_separated, phase_decision_metrics = detect_phase_separation_logic(detections, H, W, cap_bottom_y)

    if use_turbidity_decide:
        # turbidity analysis with cap exclusion
        if not is_phase_separated and img is not None:
            is_phase_separated = detect_phase_separation_from_turbidity(
                img, cap_bottom_y
            )

    if is_phase_separated:
        result = {
            "vial_state": "phase_separated",
            "n_gel": n_gel,
            "n_stable": n_stable,
            "gel_area_frac": gel_area / liquid_area if liquid_area > 0 else 0,
            "phase_separation_metrics": analyze_phase_separation(detections, H),
            "phase_separation_decision_metrics": phase_decision_metrics,
        }
        if cap_info:
            result["cap_info"] = cap_info
        return result

    # Distinguish between gelled vs stable
    gel_frac = gel_area / liquid_area if liquid_area > 0 else 0

    is_gelled = (
            gel_frac >= 0.35 or  # GEL_AREA_FRAC_THR
            (n_gel - n_stable) >= 1  # GEL_DOMINANCE_COUNT_THR
    )

    final_state = "gelled" if is_gelled else "stable"

    # Calculate liquid volume below cap
    if cap_bottom_y:
        # Adjust volume calculation to exclude cap region
        volume_estimates = calculate_liquid_volumes(detections, vial_height=H, vial_width=W, cap_bottom_y=cap_bottom_y)
    else:
        volume_estimates = calculate_liquid_volumes(detections, H, W)

    result = {
        "vial_state": final_state,
        "gel_area_frac": float(gel_frac),
        "n_gel": n_gel,
        "n_stable": n_stable,
        "liquid_coverage": liquid_area / (W * H),
        "volume_estimates_ml": volume_estimates
    }

    if cap_info:
        result["cap_info"] = cap_info

    return result

# def detect_phase_separation_from_turbidity(img, cap_bottom_y=None,
#                                            top_exclude=0.25,
#                                            bottom_exclude=0.05):
#     """
#     Detect phase separation from turbidity profile analysis.
#     Uses cap-aware exclusion if available.
#     """
#     try:
#         # Get turbidity profile with exclusions
#         v_full, v_full_norm, excluded_info = compute_turbidity_profile_with_exclusion(
#             img,
#             cap_bottom_y=cap_bottom_y,
#             top_exclude_fraction=top_exclude,
#             bottom_exclude_fraction=bottom_exclude
#         )
#
#         # Extract only the nonzero analysis region
#         middle_region = v_full_norm[np.nonzero(v_full_norm)]
#
#         if len(middle_region) < 50:  # minimum profile length
#             return False
#
#         # Gradient analysis
#         gradient = np.abs(np.gradient(middle_region))
#         threshold = np.mean(gradient) + 2.5 * np.std(gradient)
#         threshold = max(threshold, 0.06)
#
#         # Peak detection
#         significant_peaks = gradient > threshold
#         peak_positions = np.where(significant_peaks)[0]
#
#         if len(peak_positions) < 3:
#             return False
#
#         # Group nearby peaks (within % of analysis region length)
#         min_separation = len(middle_region) * 0.08
#         peak_groups = []
#         current_group = [peak_positions[0]]
#
#         for i in range(1, len(peak_positions)):
#             if peak_positions[i] - peak_positions[i - 1] < min_separation:
#                 current_group.append(peak_positions[i])
#             else:
#                 peak_groups.append(current_group)
#                 current_group = [peak_positions[i]]
#         peak_groups.append(current_group)
#
#         # Require ≥2 substantial groups of peaks
#         substantial_groups = [g for g in peak_groups if len(g) >= 2]
#         return len(substantial_groups) >= 2
#
#     except Exception as e:
#         print(f"Turbidity analysis failed: {e}")
#         return False

def calculate_liquid_volumes_with_exclusion(detections, effective_height, vial_width,
                                            cap_bottom_y, vial_volume_ml=1):
    """
    Calculate liquid volumes excluding cap region.
    """
    volumes = {}
    # Only consider area below cap
    effective_area = effective_height * vial_width

    for detection in detections:
        if is_liquid_cls(detection['class_id']):
            # Only count if detection is below cap
            if detection['box'][1] >= cap_bottom_y:
                # Adjust area calculation
                adjusted_area = detection['area']
                area_fraction = adjusted_area / effective_area
                volume_ml = area_fraction * vial_volume_ml * 0.85  # Adjustment factor

                class_name = 'gel' if detection['class_id'] == GEL_ID else 'stable'
                if class_name not in volumes:
                    volumes[class_name] = 0.0
                volumes[class_name] += volume_ml

    return volumes


# ---- Turbidity Analysis ---- #

def compute_turbidity_profile_with_exclusion(crop_bgr, cap_bottom_y=None,
                                             top_exclude_fraction=0.25,
                                             bottom_exclude_fraction=0.05):
    """
    Compute turbidity profile with region exclusion.

    Args:
        crop_bgr: BGR image of the crop
        cap_bottom_y: Y-coordinate of cap bottom (if detected)
        top_exclude_fraction: Default fraction to exclude from top if no cap
        bottom_exclude_fraction: Fraction to exclude from bottom

    Returns:
        tuple: (v_raw, v_norm, excluded_regions_info)
    """
    # Resize to standard size for analysis
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h_original, w_original = crop_bgr.shape[:2]

    # Standard analysis size
    analysis_width = 100
    analysis_height = 500
    turb = cv2.resize(crop_rgb, (analysis_width, analysis_height))

    # Convert to HSV and extract Value channel
    hsv = cv2.cvtColor(turb, cv2.COLOR_RGB2HSV)
    v_full = np.mean(hsv[:, :, -1], axis=-1)  # (500,)

    # Calculate exclusion regions
    if cap_bottom_y is not None:
        # Scale cap position to analysis dimensions
        cap_bottom_scaled = int((cap_bottom_y / h_original) * analysis_height)
        # Add small buffer below cap
        top_exclude_idx = min(cap_bottom_scaled + 10, analysis_height // 3)
    else:
        # Use default top exclusion
        top_exclude_idx = int(analysis_height * top_exclude_fraction)

    # Bottom exclusion (vial bottom artifacts)
    bottom_exclude_idx = int(analysis_height * (1 - bottom_exclude_fraction))

    # Extract analysis region
    v_analysis = v_full[top_exclude_idx:bottom_exclude_idx]

    # Normalize the analysis region
    if len(v_analysis) > 0:
        v_norm = (v_analysis - v_analysis.min()) / max(1e-6, (v_analysis.max() - v_analysis.min()))
    else:
        v_norm = np.array([])

    # Create full profile with excluded regions marked
    v_full_norm = np.zeros_like(v_full)
    if len(v_norm) > 0:
        v_full_norm[top_exclude_idx:bottom_exclude_idx] = v_norm

    excluded_info = {
        'top_exclude_idx': top_exclude_idx,
        'bottom_exclude_idx': bottom_exclude_idx,
        'analysis_height': analysis_height,
        'excluded_top_fraction': top_exclude_idx / analysis_height,
        'excluded_bottom_fraction': (analysis_height - bottom_exclude_idx) / analysis_height,
        'cap_detected': cap_bottom_y is not None
    }

    return v_full, v_full_norm, excluded_info

def save_turbidity_plot(path, v_norm, dir=None, suffix=""):
    """Save turbidity profile plot for a given normalized vector."""
    z = np.linspace(0, 1, len(v_norm))
    if dir is not None:
        filename = Path(path).stem + f"{suffix}.turbidity.png"
        out_path = Path(dir) / filename
    else:
        out_path = Path(path).with_name(Path(path).stem + f"{suffix}.turbidity.png")

    plt.figure(figsize=(3.0, 4.0), dpi=120)
    plt.plot(v_norm, z)
    plt.gca().invert_yaxis()
    plt.xlabel("Turbidity (norm)")
    plt.ylabel("Normalized Height")
    plt.title(Path(path).name + suffix, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return str(out_path)

def compute_turbidity_profile(crop):
    """Compute row-wise normalized brightness (turbidity) for a crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    v = np.mean(gray, axis=1)               # mean brightness per row
    v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)  # normalize 0–1
    return v_norm

def save_enhanced_turbidity_plot(path, v_norm, excluded_info, dir=None):
    z = np.linspace(0, 1, len(v_norm))

    if dir is not None:
        filename = Path(path).stem + ".turbidity_enhanced.png"
        out_path = dir / filename
    else:
        out_path = Path(path).with_suffix(".turbidity_enhanced.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), dpi=120)

    # Plot 1: Turbidity profile
    ax1.plot(v_norm, z, 'b-', linewidth=2)
    if excluded_info:
        ax1.axhspan(0, excluded_info['excluded_top_fraction'], alpha=0.2, color='red')
        ax1.axhspan(1 - excluded_info['excluded_bottom_fraction'], 1, alpha=0.2, color='orange')
    ax1.invert_yaxis()
    ax1.set_xlabel("Turbidity (normalized)")
    ax1.set_ylabel("Normalized Height")
    ax1.set_title("Turbidity Profile", fontsize=10)

    # Plot 2: Gradient analysis
    gradient = np.abs(np.gradient(v_norm))
    z_grad = np.linspace(0, 1, len(gradient))
    ax2.plot(gradient, z_grad, 'g-', linewidth=2)

    threshold = max(np.mean(gradient) + 2.5*np.std(gradient), 0.06)
    ax2.axvline(threshold, color='r', linestyle='--', alpha=0.5)

    # --- peak grouping ---
    peak_positions = np.where(gradient > threshold)[0]
    if len(peak_positions) > 0:
        min_sep = int(len(v_norm) * 0.1)
        groups = []
        current = [peak_positions[0]]
        for i in range(1, len(peak_positions)):
            if peak_positions[i] - peak_positions[i-1] < min_sep:
                current.append(peak_positions[i])
            else:
                groups.append(current)
                current = [peak_positions[i]]
        groups.append(current)

        # pick the strongest peak from each group
        merged_peaks = [g[np.argmax(gradient[g])] for g in groups]

        # Scatter only merged peaks
        ax2.scatter(gradient[merged_peaks], z_grad[merged_peaks],
                    c='red', s=25, label='Significant peaks')

    ax2.invert_yaxis()
    ax2.set_xlabel("Gradient Magnitude")
    ax2.set_ylabel("Normalized Height")
    ax2.set_title("Gradient Analysis", fontsize=10)

    plt.suptitle(Path(path).name, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    return str(out_path)

# ---- Manifest files ---- #

def process_and_save_manifest(crop_records, liquid_out_dir, final_manifest_path):
    """
    Process all crop records and create final manifest with all information.
    """
    liquid_out_dir = Path(liquid_out_dir)
    labels_dir = liquid_out_dir / "labels"

    processed_count = 0
    state_counts = {
        "stable": 0,
        "gelled": 0,
        "phase_separated": 0,
        "only_air": 0,
        "unknown": 0
    }
    cap_detection_count = 0

    # Ensure output directory exists
    final_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(final_manifest_path, "w") as fout:
        for rec in crop_records:
            crop_path = Path(rec["crop_image"])
            label_path = labels_dir / f"{crop_path.stem}.txt"

            # Check for alternative label file locations
            if not label_path.exists():
                alts = sorted(liquid_out_dir.parent.glob(f"exp*/labels/{crop_path.stem}.txt"))
                if alts:
                    label_path = alts[-1]

            # Use enhanced classification with cap detection
            state_info = enhanced_decide(crop_path, label_path)

            # Track cap detections
            if state_info.get("cap_info", {}).get("detected", False):
                cap_detection_count += 1

            # Enhanced turbidity analysis
            img = cv2.imread(str(crop_path))
            if img is not None:
                # Create turbidity directory
                turbidity_dir = liquid_out_dir / "turbidity"
                turbidity_dir.mkdir(parents=True, exist_ok=True)

                # Extract cap info for turbidity analysis
                cap_bottom_y = None
                if state_info.get("cap_info", {}).get("detected", False):
                    cap_bottom_y = state_info["cap_info"]["bottom_y"]

                # Compute turbidity with exclusion
                v_raw, v_norm, excluded_info = compute_turbidity_profile_with_exclusion(
                    img, cap_bottom_y
                )

                # Save enhanced plot
                plot_path = save_enhanced_turbidity_plot(
                    crop_path, v_norm, excluded_info, turbidity_dir
                )

                state_info.update({
                    "turbidity_plot": plot_path,
                    "turbidity_mean": float(np.mean(v_norm)) if len(v_norm) > 0 else 0.0,
                    "turbidity_maxstep": float(np.max(np.abs(np.diff(v_norm)))) if len(v_norm) > 1 else 0.0,
                    "turbidity_excluded_regions": excluded_info
                })

            # Combine all information
            rec.update(state_info)
            fout.write(json.dumps(rec) + "\n")

            state = state_info.get("vial_state", "unknown")
            state_counts[state] = state_counts.get(state, 0) + 1
            processed_count += 1

    # Print summary
    print(f"\n=== Processing Summary ===")
    print(f"Total vials processed: {processed_count}")
    print(f"Caps detected: {cap_detection_count} ({cap_detection_count / processed_count * 100:.1f}%)")
    print("\nState Distribution:")
    for state, count in state_counts.items():
        if count > 0:
            percentage = (count / processed_count) * 100
            print(f"  {state}: {count} ({percentage:.1f}%)")

    print(f"\nManifest saved to: {final_manifest_path}")
    return final_manifest_path

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

# ---- Main ---- #

def main():
    args = parse_args()

    # Stage A: Detect vials and create crops
    crops_dir, crop_records, temp_manifest_path, is_temp = run_stage_a_detect_and_crop(args, manifest_path=None)

    # Stage B: Run liquid detection
    liquid_out = run_stage_b_liquid(args, crops_dir)

    # Define final manifest location
    manifest_dir = liquid_out / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Extract experiment name
    exp_name = crops_dir.name
    final_manifest_path = manifest_dir / f"manifest_full_{exp_name}.jsonl"

    # Process all records and create final manifest
    manifest_with_state = process_and_save_manifest(
        crop_records,
        liquid_out,
        final_manifest_path,
    )

    # # Create summary visualization
    # summary_path = create_region_turbidity_summary(
    #     manifest_with_state,
    #     liquid_out / "turbidity"
    # )

    # Clean up temporary manifest if created
    if is_temp and temp_manifest_path.exists():
        temp_manifest_path.unlink()

    print("\n=== Pipeline Complete ===")
    print(f"Crops:       {crops_dir}")
    print(f"Liquid out:  {liquid_out}")
    print(f"Final manifest: {manifest_with_state}")
    print(f"Turbidity summary: {summary_path}")

if __name__ == "__main__":
    main()
