# ==========================================
# Image processing & computer vision
# ==========================================
import cv2                 # OpenCV for image decoding & processing
import numpy as np         # Numerical computations
import matplotlib.pyplot as plt  # Debug / visualization (optional)

# ==========================================
# Deep learning frameworks
# ==========================================
import tensorflow as tf            # TensorFlow (MoveNet backend)
import tensorflow_hub as hub       # Load pretrained MoveNet model
import torch                       # PyTorch (CLIP backend)
import clip                        # OpenAI CLIP model

# ==========================================
# Networking & image I/O
# ==========================================
import urllib.request      # Download images via URL
import requests            # HTTP requests with headers / retries
from PIL import Image      # Image handling (CLIP expects PIL)
from io import BytesIO     # Convert bytes to file-like object

# ==========================================
# Data handling & utilities
# ==========================================
import pandas as pd        # DataFrames & tabular processing
import math                # Angle / geometry calculations
import os                  # File & path handling
import time                # Timing / sleeps
from datetime import datetime  # Timestamps

# ==========================================
# Parallel processing
# ==========================================
from concurrent.futures import ThreadPoolExecutor, as_completed  # Multithreading
from tqdm import tqdm      # Progress bars


# ==========================================
# HTTP headers for image download
# (Custom User-Agent + Akamai access key)
# ==========================================
headers = {
    "User-Agent": "so5_imagescan",
    "x-akamai-secret": "dd812834-4421-4a2c-bc9a-3e93d76432b4"
}

# ==========================================
# Load MoveNet pose estimation model
# (Single-person, fast Lightning variant)
# ==========================================
movenet_model = hub.load(
    "https://tfhub.dev/google/movenet/singlepose/lightning/4"
)

# ==========================================
# Select computation device for CLIP
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Load CLIP model and preprocessing pipeline
# ==========================================
clip_model, clip_preprocess = clip.load(
    "ViT-B/32",     # Vision Transformer backbone
    device=device
)

# Set CLIP to inference mode
clip_model.eval()


def load_image_with_headers(url):
    """
    Download image from URL using custom HTTP headers
    and decode it into an OpenCV image.
    """
    # Build HTTP request with required headers
    req = urllib.request.Request(url, headers=headers)

    # Fetch raw image bytes
    resp = urllib.request.urlopen(req)
    data = resp.read()

    # Convert bytes to NumPy array
    img_arr = np.asarray(bytearray(data), dtype=np.uint8)

    # Decode image into OpenCV BGR format
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)


def get_movenet_keypoints(url):
    """
    Run MoveNet pose estimation on a single image URL.

    Returns:
        NumPy array of shape (17, 3)
        Each keypoint contains (y, x, confidence score)
    """
    # Load image from URL
    img = load_image_with_headers(url)

    # Validate image download
    if img is None:
        raise ValueError("Image download failed")

    # Convert BGR (OpenCV) to RGB (TensorFlow)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to MoveNet expected input size
    img = cv2.resize(img, (192, 192))

    # Convert to TensorFlow tensor and add batch dimension
    img = tf.cast(img, tf.int32)[tf.newaxis, ...]

    # Run MoveNet inference
    outputs = movenet_model.signatures["serving_default"](img)

    # Extract keypoints:
    # [person_index=0, keypoint_index=0, 17 keypoints, 3 values]
    return outputs["output_0"].numpy()[0, 0]


def process_image_url(url):
    """
    Process a single image URL:
    - Run MoveNet pose detection
    - Engineer pose-based features
    - Return a one-row DataFrame
    """
    try:
        # Run MoveNet inference to extract keypoints
        keypoints = get_movenet_keypoints(url)

        # Convert keypoints into engineered pose features (40+ features)
        features_df = feature_engineer(keypoints)

        # Attach image identifier
        features_df["url"] = url

        # Mark successful processing
        features_df["success"] = True

        return features_df

    except Exception as e:
        # Handle image download failures, timeouts, or inference errors
        return pd.DataFrame([{
            "url": url,          # Image identifier
            "success": False,    # Flag failure
            "error": str(e)      # Capture error message for debugging
        }])


# ==========================================
# MoveNet keypoint index mapping
# ==========================================
# Order matches MoveNet output (17 keypoints)
# Each keypoint contains (y, x, confidence score)

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

def get_clip_embedding(url, timeout=15):
    """
    Generate CLIP image embedding for a single image URL.

    Returns:
        512-dim normalized embedding vector.
        Returns zero-vector if download or inference fails.
    """
    try:
        # Download image with headers and timeout
        r = requests.get(url, headers=headers, timeout=timeout)

        # Load image into PIL and ensure RGB format
        img = Image.open(BytesIO(r.content)).convert("RGB")

        # Preprocess image and move to device (CPU / GPU)
        img = clip_preprocess(img).unsqueeze(0).to(device)

        # Run CLIP inference without gradient tracking
        with torch.no_grad():
            emb = clip_model.encode_image(img)

            # L2-normalize embedding for stability
            emb = emb / torch.clamp(
                emb.norm(dim=-1, keepdim=True),
                min=1e-6
            )

        # Return flattened NumPy vector
        return emb.cpu().numpy().flatten()

    except Exception:
        # Fallback: return zero embedding on failure
        return np.zeros(512, dtype=np.float32)


def run_clip_on_df(df, url_col="image_url", max_workers=3):
    """
    Generate CLIP embeddings for all images in a dataframe
    using multithreading.

    Returns:
        NumPy array of shape (N, 512)
    """
    clip_features = []

    # Parallelize CLIP inference across URLs
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_clip_embedding, url): url
            for url in df[url_col]
        }

        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures)):
            clip_features.append(future.result())

    # Stack all embeddings into a single array
    return np.vstack(clip_features)   # shape: (N, 512)


def keypoints_to_df(url, kp):
    """
    Convert raw MoveNet keypoints into a labeled DataFrame row.

    Args:
        url: Image URL identifier
        kp: MoveNet keypoints array of shape (17, 3)

    Returns:
        One-row DataFrame with x, y, confidence for each keypoint
    """
    row = {"url": url}

    # Map each keypoint to a named column
    for i, name in enumerate(KEYPOINT_NAMES):
        row[f"{name}_x"] = kp[i][0]
        row[f"{name}_y"] = kp[i][1]
        row[f"{name}_conf"] = kp[i][2]

    return pd.DataFrame([row])


def run_movenet_on_df(df, url_col="image_url"):
    """
    Run MoveNet pose detection sequentially on a dataframe of image URLs.

    Args:
        df: Input dataframe containing image URLs
        url_col: Column name holding image URLs

    Returns:
        DataFrame containing keypoints and success/error flags
    """
    results = []

    for i, url in enumerate(df[url_col]):
        # Progress indicator (inline update)
        print(f"Processing {i+1}/{len(df)} → {url}", end="\r")

        try:
            # Run pose estimation
            kp = get_movenet_keypoints(url)

            # Convert keypoints to structured DataFrame
            out = keypoints_to_df(url, kp)

            # Mark successful inference
            out["success"] = True

        except Exception as e:
            # Capture failures for debugging
            out = pd.DataFrame([{
                "url": url,
                "success": False,
                "error": str(e)
            }])

        results.append(out)

    # Combine all rows into a single DataFrame
    return pd.concat(results, ignore_index=True)
    

def dist(a, b):
    """
    Compute Euclidean distance between two 2D points.

    Args:
        a, b: NumPy arrays or lists representing (x, y)

    Returns:
        Float distance between points
    """
    return float(np.linalg.norm(a - b))


def angle(p1, p2):
    """
    Compute angle (in degrees) between two points.

    Angle is measured from p1 → p2 using arctangent,
    relative to the positive x-axis.

    Args:
        p1, p2: NumPy arrays or lists representing (x, y)

    Returns:
        Angle in degrees
    """
    return float(
        math.degrees(
            math.atan2(
                p2[1] - p1[1],
                p2[0] - p1[0]
            )
        )
    )

def feature_engineer(kp):
    """Generate 40+ geometric + visibility features from MoveNet keypoints."""
    
    kp = np.array(kp, dtype=float)
    
    # Extract raw points + confidence
    p = {name: kp[i][:2] for i, name in enumerate(KEYPOINT_NAMES)}
    c = {name: kp[i][2] for i, name in enumerate(KEYPOINT_NAMES)}
    
    # Visibility flags
    vis = {k: (c[k] > 0.3) for k in KEYPOINT_NAMES}

    # Limb lengths
    left_arm = dist(p["left_shoulder"], p["left_elbow"]) + dist(p["left_elbow"], p["left_wrist"])
    right_arm = dist(p["right_shoulder"], p["right_elbow"]) + dist(p["right_elbow"], p["right_wrist"])

    left_leg = dist(p["left_hip"], p["left_knee"]) + dist(p["left_knee"], p["left_ankle"])
    right_leg = dist(p["right_hip"], p["right_knee"]) + dist(p["right_knee"], p["right_ankle"])

    # Widths + torso length
    shoulder_width = dist(p["left_shoulder"], p["right_shoulder"])
    hip_width = dist(p["left_hip"], p["right_hip"])

    torso_length = dist(
        (p["left_shoulder"] + p["right_shoulder"]) / 2,
        (p["left_hip"] + p["right_hip"]) / 2
    )
    shoulder_mid = (p["left_shoulder"] + p["right_shoulder"]) / 2
    hip_mid = (p["left_hip"] + p["right_hip"]) / 2
    torso_dx = shoulder_mid[0] - hip_mid[0]
    torso_dx_norm = torso_dx / shoulder_width if shoulder_width > 0 else np.nan

    upper_conf = np.mean([
    c["left_shoulder"], c["right_shoulder"],
    c["left_elbow"], c["right_elbow"],
    c["left_wrist"], c["right_wrist"]
    ])

    lower_conf = np.mean([
        c["left_hip"], c["right_hip"],
        c["left_knee"], c["right_knee"],
        c["left_ankle"], c["right_ankle"]
    ])


    # Visibility counts
    visible_keypoints = sum(vis.values())

    visible_face = sum(vis[k] for k in ["nose", "left_eye", "right_eye", "left_ear", "right_ear"])
    
    visible_upper = sum(vis[k] for k in [
        "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist"])
    
    visible_lower = sum(vis[k] for k in [
        "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"])
    
    if vis["left_ankle"] or vis["right_ankle"]:
        lowest_visible_joint = 3
    
    elif vis["left_knee"] or vis["right_knee"]:
        lowest_visible_joint = 2
    
    elif vis["left_hip"] or vis["right_hip"]:
        lowest_visible_joint = 1
    
    elif vis["left_shoulder"] or vis["right_shoulder"]:
        lowest_visible_joint = 0
    
    else:
        lowest_visible_joint = -1

    pose_completeness = visible_keypoints / len(KEYPOINT_NAMES)

    # Angles
    shoulder_angle = angle(p["left_shoulder"], p["right_shoulder"])
    shoulder_angle = shoulder_angle if visible_upper >= 2 else np.nan
    hip_angle = angle(p["left_hip"], p["right_hip"])
    hip_angle = hip_angle if visible_lower >= 2 else np.nan
    torso_angle = angle(
        (p["left_shoulder"] + p["right_shoulder"]) / 2,
        (p["left_hip"] + p["right_hip"]) / 2
    )
    if not np.isnan(shoulder_angle) and not np.isnan(hip_angle):
        shoulder_hip_angle_diff = np.abs(shoulder_angle - hip_angle)
    else:
        shoulder_hip_angle_diff = np.nan

    upper_lr_conf_imbalance = np.abs(
        (c["left_shoulder"] + c["left_elbow"] + c["left_wrist"]) -
        (c["right_shoulder"] + c["right_elbow"] + c["right_wrist"])
    )

    lower_lr_conf_imbalance = np.abs(
        (c["left_hip"] + c["left_knee"] + c["left_ankle"]) -
        (c["right_hip"] + c["right_knee"] + c["right_ankle"])
    )
    width_collapse_ratio = shoulder_width / (hip_width + 1e-6)

    torso_vector_angle = np.arctan2(
    shoulder_mid[1] - hip_mid[1],
    shoulder_mid[0] - hip_mid[0])
    # Face visibility score
    face_vis_score = float(np.mean([
        c["nose"], c["left_eye"], c["right_eye"], c["left_ear"], c["right_ear"]
    ]))

    # Center of mass
    xs = [p[k][0] for k in KEYPOINT_NAMES if vis[k]]
    ys = [p[k][1] for k in KEYPOINT_NAMES if vis[k]]
    body_center_x = float(np.mean(xs)) if xs else np.nan
    body_center_y = float(np.mean(ys)) if ys else np.nan
    vertical_span = (max(ys) - min(ys)) if ys else np.nan

    # Ratios
    shoulder_hip_ratio = shoulder_width / hip_width if hip_width > 0 else np.nan
    arm_symmetry = left_arm / right_arm if right_arm > 0 else np.nan
    leg_symmetry = left_leg / right_leg if right_leg > 0 else np.nan

    shoulder_torso_ratio = shoulder_width / torso_length if torso_length > 0 else np.nan
    hip_torso_ratio = hip_width / torso_length if torso_length > 0 else np.nan

    

    # Build feature dict
    feats = {
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "torso_length": torso_length,

        "shoulder_angle": shoulder_angle,
        "hip_angle": hip_angle,
        "torso_angle": torso_angle,

        "left_arm_length": left_arm,
        "right_arm_length": right_arm,
        "left_leg_length": left_leg,
        "right_leg_length": right_leg,

        "shoulder_hip_ratio": shoulder_hip_ratio,
        "arm_symmetry": arm_symmetry,
        "leg_symmetry": leg_symmetry,
        "shoulder_torso_ratio": shoulder_torso_ratio,
        "hip_torso_ratio": hip_torso_ratio,

        "body_center_x": body_center_x,
        "body_center_y": body_center_y,

        "visibility_face_score": face_vis_score,
        "visible_keypoints": visible_keypoints,
        "visible_face_keypoints": visible_face,
        "visible_upper_body": visible_upper,
        "visible_lower_body": visible_lower,
        "vertical_span": vertical_span,
        "lowest_visible_joint": lowest_visible_joint,
        "torso_dx": torso_dx,
        "torso_dx_norm":torso_dx_norm,
        "mean_upper_conf": upper_conf,
        "mean_lower_conf": lower_conf,
        "conf_upper_minus_lower": upper_conf - lower_conf,
        "pose_completeness": pose_completeness,
        "shoulder_hip_angle_diff": shoulder_hip_angle_diff,
        "upper_lr_conf_imbalance": upper_lr_conf_imbalance,
        "lower_lr_conf_imbalance": lower_lr_conf_imbalance,
        "width_collapse_ratio": width_collapse_ratio,
        "torso_vector_angle": torso_vector_angle
    }

    # Add raw keypoints (x,y,conf + visibility)
    for k in KEYPOINT_NAMES:
        feats[f"{k}_x"] = float(p[k][0])
        feats[f"{k}_y"] = float(p[k][1])
        feats[f"{k}_conf"] = float(c[k])
        feats[f"vis_{k}"] = vis[k]

    return pd.DataFrame([feats])

def extract_kp_from_row(row):
    """
    Reconstruct MoveNet keypoints array from a dataframe row.

    Args:
        row: Pandas Series containing *_x, *_y, *_conf columns

    Returns:
        NumPy array of shape (17, 3)
    """
    # Select all keypoint-related columns
    cols = [c for c in row.index if c.endswith(("_x", "_y", "_conf"))]

    # Convert flat values back to (17, 3) keypoint structure
    kp = row[cols].values.astype(float).reshape(17, 3)

    return kp


def run_feature_engineering(df):
    """
    Run pose feature engineering on a dataframe containing
    MoveNet keypoints.

    Args:
        df: DataFrame with keypoints and success flag

    Returns:
        DataFrame with engineered pose features
    """
    all_features = []

    for _, row in df.iterrows():
        # Skip rows where pose inference failed
        if not row["success"]:
            continue

        # Reconstruct keypoints from stored columns
        kp = extract_kp_from_row(row)

        # Generate pose-based features
        feats = feature_engineer(kp)

        # Attach image identifier
        feats["url"] = row["url"]

        all_features.append(feats)

    # Combine all feature rows into a single DataFrame
    return pd.concat(all_features, ignore_index=True)


def build_final_feature_df(movenet_df):
    """
    Build final pose-based feature dataframe from MoveNet output.

    Args:
        movenet_df: DataFrame containing MoveNet keypoints,
                    success flag, and image URLs

    Returns:
        DataFrame with engineered pose features + URL
    """
    final_rows = []

    for _, row in movenet_df.iterrows():

        # Skip rows where MoveNet inference failed
        if not row.get("success", True):
            continue

        # Reconstruct 17x3 MoveNet keypoints from stored columns
        kp = extract_kp_from_row(row)

        # Generate engineered geometric and visibility features
        feats = feature_engineer(kp)

        # Attach image identifier (used for merges downstream)
        feats["url"] = row["url"]

        final_rows.append(feats)

    # Combine all feature rows into a single DataFrame
    return pd.concat(final_rows, ignore_index=True)


SKELETON_EDGES = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

SKELETON_EDGES = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

def draw_keypoints_on_image(img, keypoints):
    h, w, _ = img.shape

    # Convert to dict {name: (x,y,conf)}
    pts = {
        KEYPOINT_NAMES[i]: keypoints[i]
        for i in range(17)
    }

    # --- Draw skeleton lines ---
    for a, b in SKELETON_EDGES:
        xa, ya, ca = pts[a]
        xb, yb, cb = pts[b]

        if ca > 0.2 and cb > 0.2:  # both points must be visible
            cv2.line(
                img,
                (int(xa * w), int(ya * h)),
                (int(xb * w), int(yb * h)),
                (0, 255, 0), 2
            )

    # --- Draw keypoint dots ---
    for name, (x, y, c) in pts.items():
        if c > 0.2:
            cv2.circle(
                img,
                (int(x * w), int(y * h)),
                4,
                (0, 0, 255),
                -1
            )

    return img

def draw_keypoints_on_image_conf(img, keypoints):
    h, w, _ = img.shape

    pts = {
        KEYPOINT_NAMES[i]: keypoints[i] 
        for i in range(17)
    }

    for a, b in SKELETON_EDGES:
        xa, ya, ca = pts[a]
        xb, yb, cb = pts[b]

        # edge confidence = min of both endpoints
        conf = min(ca, cb)
        conf = max(0.0, min(conf, 1.0))

        # scale color by confidence
        color = (0, int(255 * conf), 0)   # green fades when conf low
        thickness = 2 if conf > 0.2 else 1

        cv2.line(
            img,
            (int(xa * w), int(ya * h)),
            (int(xb * w), int(yb * h)),
            color,
            thickness
        )

    # draw keypoints
    for name, (x, y, c) in pts.items():
        conf = max(0.0, min(c, 1.0))
        color = (0, 0, int(255 * conf))  # red fades with low conf
        radius = 5 if conf > 0.3 else 3

        cv2.circle(
            img,
            (int(x * w), int(y * h)),
            radius,
            color,
            -1
        )

    return img    

def draw_keypoints_force_visible(img, keypoints):
    h, w, _ = img.shape
    pts = {KEYPOINT_NAMES[i]: keypoints[i] for i in range(17)}

    for a, b in SKELETON_EDGES:
        xa, ya, ca = pts[a]
        xb, yb, cb = pts[b]
        conf = max(min(min(ca, cb), 1.0), 0.0)

        # FORCE MINIMUM VISIBLE COLOR (so even low conf shows)
        brightness = int(60 + conf * 195)  # ranges from 60→255
        color = (0, brightness, 0)

        cv2.line(
            img,
            (int(xa*w), int(ya*h)),
            (int(xb*w), int(yb*h)),
            color,
            2
        )

    for name, (x, y, c) in pts.items():
        brightness = int(60 + c * 195)
        color = (0, 0, brightness)

        cv2.circle(
            img,
            (int(x*w), int(y*h)),
            4,
            color,
            -1
        )

    return img



def load_image_with_headers(url):
    """Download image with required headers."""
    req = urllib.request.Request(url, headers=headers)
    resp = urllib.request.urlopen(req, timeout=10)  # timeout protection
    data = resp.read()

    img_arr = np.asarray(bytearray(data), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    return img


def get_movenet_keypoints(url):
    """Download image → preprocess → run MoveNet → return keypoints."""
    
    img = load_image_with_headers(url)   # uses your headers
    if img is None:
        raise ValueError("Failed to load image.")

    # Preprocess for MoveNet
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = tf.image.resize(img_rgb, (192, 192))
    input_tensor = tf.cast(img_resized, dtype=tf.int32)[tf.newaxis, ...]

    # Inference
    outputs = movenet_model.signatures["serving_default"](input_tensor)
    keypoints = outputs["output_0"].numpy()[0, 0]  # shape: (17, 3)

    return keypoints

def keypoints_to_row(url, kp):
    """Turn MoveNet 17 keypoints into a labeled dict for DF."""
    
    row = {"url": url}

    for i, name in enumerate(KEYPOINT_NAMES):
        row[f"{name}_x"] = float(kp[i][0])
        row[f"{name}_y"] = float(kp[i][1])
        row[f"{name}_conf"] = float(kp[i][2])

    row["success"] = True
    return row

def safe_get_keypoints(url, retries=3, delay=1):
    for attempt in range(retries):
        try:
            return get_movenet_keypoints(url)
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(delay)


def process_one_url(url):
    """
    Process a single image URL for MoveNet pose extraction.

    Steps:
    - Safely download image and extract keypoints
    - Convert keypoints into a structured output row
    - Handle and record failures gracefully
    """
    try:
        # Safely run MoveNet keypoint extraction (with internal error handling)
        kp = safe_get_keypoints(url)

        # Convert keypoints into a flat row format
        return keypoints_to_row(url, kp)

    except Exception as e:
        # Capture any failure (download / inference / timeout)
        return {
            "url": url,
            "success": False,
            "error": str(e)
        }


def run_movenet_multithread(df, url_col="image_url", max_workers=8, timeout=15):
    """
    Run MoveNet pose extraction in parallel for a dataframe of image URLs.

    Args:
        df: Input dataframe containing image URLs
        url_col: Column name holding image URLs
        max_workers: Number of parallel threads
        timeout: Maximum time allowed per image (seconds)

    Returns:
        DataFrame containing pose features, URL, and success/error flags
    """
    results = []

    # Extract list of image URLs
    urls = df[url_col].tolist()

    # Parallel execution using thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one_url, url): url
            for url in urls
        }

        # Collect results as they complete
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing"
        ):
            url = futures[future]

            try:
                # Retrieve result with timeout handling
                result = future.result(timeout=timeout)
            except Exception as e:
                # Capture failures (timeouts, download errors, inference errors)
                result = {
                    "url": url,
                    "success": False,
                    "error": str(e)
                }

            results.append(result)

    # Combine all per-image outputs into a DataFrame
    return pd.DataFrame(results)



def save_outputs(df, base_name="movenet_output"):
    """
    Save dataframe output to a timestamped parquet file.

    Args:
        df: DataFrame to persist
        base_name: Prefix for output filename
    """
    # Generate timestamp (DDMMYYYYHHMMSS)
    ts = datetime.now().strftime("%d%m%Y%H%M%S")

    # Construct output filename
    parquet_name = f"{base_name}_{ts}.parquet"

    # Save dataframe as parquet
    df.to_parquet(parquet_name, index=False)

    # Log saved file
    print("Saved files:")
    print(" →", parquet_name)