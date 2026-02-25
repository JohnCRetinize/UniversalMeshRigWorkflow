import os
import json
import random
import subprocess
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import folder_paths
from .install_manager import AutoRigInstaller

# Define Node Root
NODE_ROOT = Path(__file__).parent
TEMP_DIR = NODE_ROOT / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSAL_RIG_TEMPLATE = Path(r"C:\Users\jack\Downloads\AI2Mesh_CCUniversalRig_NewHierarchy.fbx")

# Load CC_Base mapping preset (vendored) for universal rig mapping
def _load_cc_map():
    path = NODE_ROOT / "templates" / "cc_map.bmap"
    entries = []
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Each entry spans multiple lines in .bmap; lines that look like 'CC_Base_*' are names
                    if line.startswith("CC_Base_"):
                        entries.append(line)
            print(f"### [AutoRig] Loaded CC_Base mapping preset with {len(entries)} entries from {path}")
        except Exception as e:
            print(f"### [AutoRig] Failed to load CC_Base mapping: {e}")
    else:
        print(f"### [AutoRig] CC_Base mapping file not found at {path}")
    return entries

CC_BASE_PRESET = _load_cc_map()

def synthesize_missing_cc_bones(skel, preset=CC_BASE_PRESET):
    """Ensure all CC_Base entries from the preset exist in the skeleton.
    Missing entries are added with a default zero position and returned along with a list of added names.
    """
    added = []
    for name in preset:
        if name not in skel:
            skel[name] = [0.0, 0.0, 0.0]
            added.append(name)
    return skel, added



# --------------------------------------------------------------------------------
# RTMPose / OpenPose MAPPING (Verified)
# --------------------------------------------------------------------------------
OPENPOSE_MAP = {
    "0": "Nose", "1": "Neck",
    "2": "Shoulder.R", "3": "Arm.R", "4": "Forearm.R",
    "5": "Shoulder.L", "6": "Arm.L", "7": "Forearm.L",
    "8": "Thigh.R", "9": "Shin.R", "10": "Ankle.R",
    "11": "Thigh.L", "12": "Shin.L", "13": "Ankle.L",
    "14": "Eye.R", "15": "Eye.L",
    "16": "Ear.R", "17": "Ear.L",
    "18": "BigToe.L",
    "19": "SmallToe.L",
    "20": "Heel.L",
    "21": "BigToe.R",
    "22": "SmallToe.R",
    "23": "Heel.R"
}

# DWPose face keypoints (68 points from face_keypoints_2d array)
# Indices 0-67 in the face array correspond to OpenPose 68-point face model
FACE_KEYPOINT_MAP = {
    "8": "Face.Chin",  # Chin/jaw bottom (index 8 in 68-point model)
    "62": "Mouth.Inner.Top",  # Inner mouth top
    "66": "Mouth.Inner.Bottom",  # Inner mouth bottom
}

HAND_JOINT_MAP = {
    0: "Hand",
    1: "Thumb.1", 2: "Thumb.2", 3: "Thumb.3", 4: "Thumb.Tip",
    5: "Index.1", 6: "Index.2", 7: "Index.3", 8: "Index.Tip",
    9: "Middle.1", 10: "Middle.2", 11: "Middle.3", 12: "Middle.Tip",
    13: "Ring.1", 14: "Ring.2", 15: "Ring.3", 16: "Ring.Tip",
    17: "Pinky.1", 18: "Pinky.2", 19: "Pinky.3", 20: "Pinky.Tip"
}

# --------------------------------------------------------------------------------
# ADVANCED MATH HELPERS (RANSAC Triangulation)
# --------------------------------------------------------------------------------

def procrustes_align(source_points, target_points):
    """
    Align source points to target points using Procrustes analysis.
    Returns rotation matrix, scale, and translation that best align source to target.
    
    This solves the orientation problem: triangulated hand has good shape but wrong rotation.
    We align it to the body-pass hand's orientation.
    
    Args:
        source_points: Nx3 array of triangulated positions
        target_points: Nx3 array of body-pass positions (reference orientation)
    
    Returns:
        R: 3x3 rotation matrix
        s: scalar scale
        t: 3D translation vector
    """
    # Center both point clouds
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    
    # Compute scale
    source_scale = np.sqrt(np.sum(source_centered**2) / len(source_points))
    target_scale = np.sqrt(np.sum(target_centered**2) / len(target_points))
    scale = target_scale / source_scale if source_scale > 1e-6 else 1.0
    
    # Normalize for rotation computation
    source_normalized = source_centered / (source_scale + 1e-8)
    target_normalized = target_centered / (target_scale + 1e-8)
    
    # Compute optimal rotation using SVD
    H = source_normalized.T @ target_normalized
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1, not -1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Translation after rotation and scale
    t = target_center - scale * (R @ source_center)
    
    return R, scale, t

def solve_dlt_weighted(pts, matrices, weights):
    """Linear triangulation solver."""
    A = []
    for (u, v), P, w in zip(pts, matrices, weights):
        row_u = (u * P[2] - P[0]) * w
        row_v = (v * P[2] - P[1]) * w
        A.append(row_u)
        A.append(row_v)
    A = np.array(A)
    try:
        _, _, vt = np.linalg.svd(A)
        X = vt[-1]
        return X / X[3] if X[3] != 0 else X
    except:
        return np.array([0., 0., 0., 1.0])

def project_point(X, P):
    """Projects 3D point X using matrix P back to 2D coordinates."""
    x_proj = P @ X
    if abs(x_proj[2]) < 1e-6: return -1, -1
    u = x_proj[0] / x_proj[2]
    v = x_proj[1] / x_proj[2]
    return u, v

def triangulate_ransac(keypoint_data, projection_matrices):
    """
    Uses RANSAC to ignore frames where the joint is occluded or misidentified.
    This solves the issue of the 15-photo mapping being worse than 10-photo.
    """
    # Only consider frames with reasonable confidence
    valid_idx = [i for i, kp in enumerate(keypoint_data) if kp and kp[2] > 0.15]
    if len(valid_idx) < 2: return [0.0, 0.0, 0.0]
    
    best_point = [0.0, 0.0, 0.0]
    max_inliers = -1
    
    # RANSAC iterations (Trial and Error to find the most 'voted for' 3D position)
    for _ in range(min(len(valid_idx) * 2, 20)):
        # Sample 2 random views
        sample_indices = random.sample(valid_idx, 2)
        
        s_pts = [( (keypoint_data[i][0]*2)-1, 1-(keypoint_data[i][1]*2) ) for i in sample_indices]
        s_mats = [np.array(projection_matrices[i]) for i in sample_indices]
        s_w = [keypoint_data[i][2] for i in sample_indices]
        
        X_candidate = solve_dlt_weighted(s_pts, s_mats, s_w)
        
        # Count inliers (views that agree with this 3D point)
        inliers = 0
        current_inlier_indices = []
        for i in valid_idx:
            u_actual = (keypoint_data[i][0] * 2) - 1
            v_actual = 1 - (keypoint_data[i][1] * 2)
            u_p, v_p = project_point(X_candidate, np.array(projection_matrices[i]))
            
            error = np.sqrt((u_actual - u_p)**2 + (v_actual - v_p)**2)
            if error < 0.05: # Inlier threshold (approx 25 pixels in 1024 space)
                inliers += 1
                current_inlier_indices.append(i)
        
        if inliers > max_inliers:
            max_inliers = inliers
            # Re-solve using all inliers for maximum precision
            f_pts = [( (keypoint_data[i][0]*2)-1, 1-(keypoint_data[i][1]*2) ) for i in current_inlier_indices]
            f_mats = [np.array(projection_matrices[i]) for i in current_inlier_indices]
            f_w = [keypoint_data[i][2] for i in current_inlier_indices]
            best_point = solve_dlt_weighted(f_pts, f_mats, f_w)[:3].tolist()
            
    return best_point

def extract_keypoint(pose_frame, joint_idx, resolution=1024):
    """
    Extract body keypoint from pose data.
    DWPose outputs keypoints in pixel coordinates, we normalize them to 0-1 range.
    """
    try:
        if isinstance(pose_frame, dict) and 'people' in pose_frame and len(pose_frame['people']) > 0:
            person = pose_frame['people'][0]
            data = person.get('pose_keypoints_2d', [])
            idx = joint_idx * 3
            if idx + 2 < len(data):
                return [data[idx] / float(resolution), data[idx+1] / float(resolution), data[idx+2]]
    except: pass
    return None

def extract_face_keypoint(pose_frame, face_idx, resolution=1024):
    """
    Extract face keypoint from pose data.
    DWPose outputs face keypoints (68 points, indices 0-67) in face_keypoints_2d array.
    
    Args:
        pose_frame: Pose data dict containing 'people' array
        face_idx: Index of the face keypoint (0-67 for 68 face points)
        resolution: Image resolution for normalization
    """
    try:
        if isinstance(pose_frame, dict) and 'people' in pose_frame and len(pose_frame['people']) > 0:
            person = pose_frame['people'][0]
            data = person.get('face_keypoints_2d', [])
            idx = face_idx * 3
            if idx + 2 < len(data):
                return [data[idx] / float(resolution), data[idx+1] / float(resolution), data[idx+2]]
    except: pass
    return None

def extract_hand_keypoint(pose_frame, hand_type, joint_idx, resolution=1024):
    """
    Extract hand keypoint from pose data.
    DWPose outputs keypoints in pixel coordinates, we normalize them to 0-1 range.
    
    Args:
        pose_frame: Pose data dict containing 'people' array
        hand_type: 'left' or 'right'
        joint_idx: Index of the joint (0-20 for 21 hand keypoints)
        resolution: Image resolution (default 1024, but hand images may use different res)
    """
    try:
        if isinstance(pose_frame, dict) and 'people' in pose_frame and len(pose_frame['people']) > 0:
            person = pose_frame['people'][0]
            key = 'hand_left_keypoints_2d' if hand_type == "left" else 'hand_right_keypoints_2d'
            data = person.get(key, [])
            idx = joint_idx * 3
            if idx + 2 < len(data):
                # Normalize pixel coordinates to 0-1 range based on actual resolution
                return [data[idx] / float(resolution), data[idx+1] / float(resolution), data[idx+2]]
    except Exception as e:
        pass
    return None

# --------------------------------------------------------------------------------
# ALIGNMENT HELPERS
# --------------------------------------------------------------------------------

def mirror_better_side(skeleton, dwpose_keypoints):
    """
    If mesh is symmetrical but one side has better detection, mirror the better side.
    Compares left/right detection quality and mirrors the side with higher confidence.
    """
    print(f"### [AutoRig] Checking for asymmetric detection quality...")
    
    # Define limb groups to check independently
    limb_groups = {
        "arm": [("Shoulder.L", "Shoulder.R"), ("Arm.L", "Arm.R"), ("Forearm.L", "Forearm.R")],
        "leg": [("Thigh.L", "Thigh.R"), ("Shin.L", "Shin.R"), ("Ankle.L", "Ankle.R")],
        "foot": [("Ankle.L", "Ankle.R"), ("BigToe.L", "BigToe.R"), ("Heel.L", "Heel.R")],
        "hand": [("Hand.L", "Hand.R")],
    }
    
    # Add finger groups to hand mirroring
    finger_pairs = []
    for finger in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
        for i in range(1, 5):  # .1, .2, .3, .Tip
            if i < 4:
                finger_pairs.append((f"{finger}.{i}.L", f"{finger}.{i}.R"))
            else:
                finger_pairs.append((f"{finger}.Tip.L", f"{finger}.Tip.R"))
    limb_groups["fingers"] = finger_pairs
    
    # Get average confidence scores for each side from DWPose data
    def get_keypoint_confidence(kp_data, keypoint_name):
        """Extract confidence from DWPose keypoints"""
        if not kp_data:
            return 0.0
        
        # Map keypoint names to indices
        name_to_idx = {v: int(k) for k, v in OPENPOSE_MAP.items()}
        if keypoint_name not in name_to_idx:
            # Check if it's a hand keypoint
            if keypoint_name in HAND_JOINT_MAP.values() or any(keypoint_name.startswith(f"{finger}.") for finger in ["Thumb", "Index", "Middle", "Ring", "Pinky"]):
                # For hand keypoints, check hand_keypoints_2d instead
                frames = kp_data if isinstance(kp_data, list) else [kp_data]
                confidences = []
                for frame in frames:
                    if 'people' in frame and len(frame['people']) > 0:
                        # Determine hand type from keypoint name
                        hand_key = 'hand_left_keypoints_2d' if keypoint_name.endswith('.L') else 'hand_right_keypoints_2d'
                        hand_kps = frame['people'][0].get(hand_key, [])
                        if len(hand_kps) > 0:
                            # Average all hand keypoint confidences
                            for i in range(0, len(hand_kps), 3):
                                if i + 2 < len(hand_kps):
                                    conf = hand_kps[i + 2]
                                    if conf > 0:
                                        confidences.append(conf)
                return np.mean(confidences) if confidences else 0.0
            return 0.0
        
        idx = name_to_idx[keypoint_name]
        
        # Average confidence across all frames
        confidences = []
        frames = kp_data if isinstance(kp_data, list) else [kp_data]
        for frame in frames:
            if 'people' in frame and len(frame['people']) > 0:
                kps = frame['people'][0].get('pose_keypoints_2d', [])
                if idx * 3 + 2 < len(kps):
                    conf = kps[idx * 3 + 2]
                    if conf > 0:
                        confidences.append(conf)
        
        return np.mean(confidences) if confidences else 0.0
    
    # Check each limb group
    for group_name, pairs in limb_groups.items():
        left_confidence = []
        right_confidence = []
        
        for left_name, right_name in pairs:
            left_conf = get_keypoint_confidence(dwpose_keypoints, left_name)
            right_conf = get_keypoint_confidence(dwpose_keypoints, right_name)
            left_confidence.append(left_conf)
            right_confidence.append(right_conf)
        
        avg_left = np.mean(left_confidence) if left_confidence else 0.0
        avg_right = np.mean(right_confidence) if right_confidence else 0.0
        
        # If one side significantly better (>15% difference), mirror it
        confidence_diff = abs(avg_left - avg_right)
        confidence_threshold = 0.15
        
        if confidence_diff > confidence_threshold:
            if avg_left > avg_right:
                # Mirror left to right
                print(f"### [AutoRig] {group_name.upper()}: Left side better (L:{avg_left:.3f} vs R:{avg_right:.3f}), mirroring L→R")
                for left_name, right_name in pairs:
                    if left_name in skeleton:
                        left_pos = skeleton[left_name]
                        # Mirror X position, keep Y and Z
                        mirrored_pos = [-left_pos[0], left_pos[1], left_pos[2]]
                        skeleton[right_name] = mirrored_pos
                        print(f"###   Mirrored {left_name} → {right_name}")
            else:
                # Mirror right to left
                print(f"### [AutoRig] {group_name.upper()}: Right side better (R:{avg_right:.3f} vs L:{avg_left:.3f}), mirroring R→L")
                for left_name, right_name in pairs:
                    if right_name in skeleton:
                        right_pos = skeleton[right_name]
                        # Mirror X position, keep Y and Z
                        mirrored_pos = [-right_pos[0], right_pos[1], right_pos[2]]
                        skeleton[left_name] = mirrored_pos
                        print(f"###   Mirrored {right_name} → {left_name}")
        else:
            print(f"### [AutoRig] {group_name.upper()}: Both sides similar quality (L:{avg_left:.3f}, R:{avg_right:.3f}), no mirroring")

def align_body_centerline(skeleton):
    """
    Align spine, neck, head on body centerline (X=0 in front view).
    Preserves Y-depth for side-view curvature.
    This ensures the body looks straight from the front.
    """
    # Bones that should be on the centerline (single plane in front view)
    centerline_bones = [
        "Hips", "Spine", "Neck", "Head", "Nose"
    ]
    
    # Calculate the average X position (centerline)
    x_positions = []
    for bone_name in centerline_bones:
        if bone_name in skeleton:
            x_positions.append(skeleton[bone_name][0])
    
    if not x_positions:
        print(f"### [AutoRig] No centerline bones found for alignment")
        return
    
    center_x = np.mean(x_positions)
    print(f"### [AutoRig] Body centerline at X={center_x:.4f}")
    
    # Align centerline bones to this X position
    for bone_name in centerline_bones:
        if bone_name in skeleton:
            old_pos = skeleton[bone_name]
            skeleton[bone_name] = [center_x, old_pos[1], old_pos[2]]
            offset = abs(old_pos[0] - center_x)
            if offset > 0.01:  # Only log significant changes
                print(f"### [AutoRig] Aligned {bone_name}: X {old_pos[0]:.4f} → {center_x:.4f} (offset {offset:.4f}m)")
    
    # Align left/right pairs symmetrically around centerline
    paired_bones = [
        ("Shoulder.L", "Shoulder.R"),
        ("Arm.L", "Arm.R"),
        ("Forearm.L", "Forearm.R"),
        ("Thigh.L", "Thigh.R"),
        ("Shin.L", "Shin.R"),
        ("Ankle.L", "Ankle.R"),
    ]
    
    for left_name, right_name in paired_bones:
        if left_name in skeleton and right_name in skeleton:
            left_pos = skeleton[left_name]
            right_pos = skeleton[right_name]
            
            # Calculate how far each should be from centerline (symmetric)
            current_left_offset = left_pos[0] - center_x
            current_right_offset = right_pos[0] - center_x
            
            # Average distance from center (should be same magnitude, opposite sign)
            avg_distance = (abs(current_left_offset) + abs(current_right_offset)) / 2
            
            # Force symmetry: left is +X, right is -X (or vice versa depending on current)
            # Determine which side is which based on current positions
            if current_left_offset > 0:  # Left is already positive X
                new_left_x = center_x + avg_distance
                new_right_x = center_x - avg_distance
            else:  # Left is negative X (flipped model)
                new_left_x = center_x - avg_distance
                new_right_x = center_x + avg_distance
            
            # Apply symmetric positions, preserve Y and Z
            skeleton[left_name] = [new_left_x, left_pos[1], left_pos[2]]
            skeleton[right_name] = [new_right_x, right_pos[1], right_pos[2]]
            
            # Log if significant change
            left_change = abs(left_pos[0] - new_left_x)
            right_change = abs(right_pos[0] - new_right_x)
            if left_change > 0.01 or right_change > 0.01:
                print(f"### [AutoRig] Symmetrized {left_name}/{right_name}: L offset {left_change:.4f}m, R offset {right_change:.4f}m")

# --------------------------------------------------------------------------------
# NODES
# --------------------------------------------------------------------------------

class UR_Triangulate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dwpose_keypoints": ("*",),
            },
            "optional": {
                "camera_metadata": ("METADATA",),
                "initial_skeleton": ("SKELETON_3D",),
                "hand_cam_metadata": ("METADATA",),
                "hand_to_refine": (["left", "right"],),
            }
        }
    RETURN_TYPES = ("SKELETON_3D",)
    FUNCTION = "exec"
    CATEGORY = "UniversalRig"
    def exec(self, dwpose_keypoints, camera_metadata=None, initial_skeleton=None, hand_cam_metadata=None, hand_to_refine=None):
        
        skel_3d = initial_skeleton.copy() if initial_skeleton else {}
        
        # Helper function to add joint kinks for better articulation
        def add_joint_kink(skeleton, joint_name, forward_direction, amount=0.05):
            """
            Add slight forward bend to joints (elbows, knees) for proper articulation.
            This prevents locked-straight limbs and allows natural bending.
            """
            if joint_name in skeleton:
                pos = skeleton[joint_name]
                skeleton[joint_name] = [
                    pos[0] + forward_direction[0] * amount,
                    pos[1] + forward_direction[1] * amount,
                    pos[2] + forward_direction[2] * amount
                ]
                print(f"### [AutoRig] Added {amount}m kink to {joint_name} → {skeleton[joint_name]}")
        
        # Hand refinement path - ensure we have matching keypoints for hand views
        if initial_skeleton and hand_cam_metadata and hand_to_refine:
            # REFINED HAND PATH
            # The hand camera frames are centered on the hand, so triangulated positions
            # are in a local coordinate system. We need to anchor them to the known wrist position.
            print(f"### [AutoRig] Refining hand triangulation for {hand_to_refine} hand...")
            
            hand_short = "L" if hand_to_refine == "left" else "R"
            suffix = f".{hand_short}"
            
            if hand_short in hand_cam_metadata:
                matrices = hand_cam_metadata[hand_short]['matrices']
                num_frames = len(matrices)
                resolution = hand_cam_metadata[hand_short].get('resolution', 512)
                hand_center_world = np.array(hand_cam_metadata[hand_short].get('center', [0, 0, 0]))
                
                # Ensure dwpose_keypoints matches the number of hand views
                if isinstance(dwpose_keypoints, dict):
                    dwpose_keypoints = [dwpose_keypoints] * num_frames
                elif isinstance(dwpose_keypoints, list):
                    if len(dwpose_keypoints) != num_frames:
                        print(f"### [AutoRig] WARNING: Hand keypoints={len(dwpose_keypoints)} but hand views={num_frames}")
                        # Pad or truncate
                        if len(dwpose_keypoints) < num_frames:
                            dwpose_keypoints = dwpose_keypoints + [dwpose_keypoints[-1]] * (num_frames - len(dwpose_keypoints))
                        else:
                            dwpose_keypoints = dwpose_keypoints[:num_frames]
                
                print(f"### [AutoRig] Using {num_frames} views at resolution {resolution}")
                
                # Triangulate the wrist to get offset for anchoring
                wrist_joint_data = [extract_hand_keypoint(dwpose_keypoints[i], hand_to_refine, 0, resolution) for i in range(num_frames)]
                valid_wrist_count = sum(1 for w in wrist_joint_data if w is not None and w[2] > 0.15)
                print(f"### [AutoRig] Wrist detected in {valid_wrist_count}/{num_frames} frames")
                
                triangulated_wrist = np.array(triangulate_ransac(wrist_joint_data, matrices))
                
                # Get the original wrist position from initial body pass
                wrist_key = f"Hand.{hand_short}"
                original_wrist = np.array(skel_3d.get(wrist_key, [0, 0, 0]))
                
                # Calculate the centroid of all triangulated points
                all_triangulated = []
                for j_idx in HAND_JOINT_MAP.keys():
                    jd = [extract_hand_keypoint(dwpose_keypoints[i], hand_to_refine, j_idx, resolution) for i in range(num_frames)]
                    tp = np.array(triangulate_ransac(jd, matrices))
                    if np.linalg.norm(tp) > 0.001:
                        all_triangulated.append(tp)
                
                if len(all_triangulated) > 0:
                    triangulated_center = np.mean(all_triangulated, axis=0)
                    
                    # Use hand_center as the primary anchor since cameras are framed on it
                    center_offset = hand_center_world - triangulated_center
                    print(f"### [AutoRig] Center-based offset: {center_offset}")
                    
                    # Also compute wrist-based offset for comparison
                    if np.linalg.norm(triangulated_wrist) > 0.001 and np.linalg.norm(original_wrist) > 0.001:
                        wrist_offset = original_wrist - triangulated_wrist
                        print(f"### [AutoRig] Wrist-based offset: {wrist_offset}")
                        print(f"### [AutoRig] Offset difference (center - wrist): {center_offset - wrist_offset}")
                    
                    # Use the center-based offset as it's more reliable
                    # (cameras are framed on center, so triangulation should be most accurate there)
                    offset = center_offset
                else:
                    # Fallback to wrist-based offset
                    if np.linalg.norm(triangulated_wrist) > 0.001 and np.linalg.norm(original_wrist) > 0.001:
                        offset = original_wrist - triangulated_wrist
                        print(f"### [AutoRig] Using wrist offset (fallback): {offset}")
                    else:
                        offset = np.array([0, 0, 0])
                        print(f"### [AutoRig] WARNING: Could not compute offset, using zero")

                # Perform refined triangulation and apply offset
                # We use a hybrid approach: anchor the wrist to its original position
                # and place fingers relative to the wrist using triangulated positions
                updated_count = 0
                
                # First, triangulate all joints
                triangulated_joints = {}
                for j_idx, b_base in HAND_JOINT_MAP.items():
                    b_name = b_base + suffix
                    joint_data = [extract_hand_keypoint(dwpose_keypoints[i], hand_to_refine, j_idx, resolution) for i in range(num_frames)]
                    valid_views = sum(1 for jd in joint_data if jd is not None and jd[2] > 0.15)
                    triangulated_pos = np.array(triangulate_ransac(joint_data, matrices))
                    if np.linalg.norm(triangulated_pos) > 0.001:
                        triangulated_joints[b_name] = {'pos': triangulated_pos, 'valid_views': valid_views, 'keypoints': joint_data}
                
                # Get triangulated wrist for relative positioning
                wrist_tri_key = f"Hand{suffix}"
                if wrist_tri_key in triangulated_joints:
                    tri_wrist_pos = triangulated_joints[wrist_tri_key]['pos']
                    
                    # Get body-pass wrist position for reference
                    body_wrist = np.array(skel_3d.get(wrist_tri_key, [0, 0, 0]))
                    
                    # Debug info
                    print(f"### [AutoRig] Coordinate Comparison:")
                    print(f"###   Body-pass wrist: {body_wrist}")
                    print(f"###   Triangulated wrist: {tri_wrist_pos}")
                    print(f"###   Hand camera center: {hand_center_world}")
                    
                    middle_tip_key = f"Middle.3{suffix}"
                    if middle_tip_key in triangulated_joints:
                        tri_middle = triangulated_joints[middle_tip_key]['pos']
                        tri_hand_span = np.linalg.norm(tri_middle - tri_wrist_pos)
                        print(f"###   Triangulated hand span: {tri_hand_span:.4f}m")
                    
                    # ANCHOR TO HAND CENTER: The cameras were all looking at hand_center_world,
                    # so that's the most accurate reference point for the triangulation
                    print(f"### [AutoRig] Using hand camera center as anchor point...")
                    
                    # Calculate where the triangulated hand's center is
                    all_tri_positions = [data['pos'] for data in triangulated_joints.values()]
                    tri_center = np.mean(all_tri_positions, axis=0)
                    
                    # Translation to move triangulated center to hand_center_world
                    translation = hand_center_world - tri_center
                    print(f"###   Translation vector: {translation}")
                    print(f"###   Translation distance: {np.linalg.norm(translation):.4f}m")
                    
                    # Apply translation to all triangulated joints
                    for b_name, data in triangulated_joints.items():
                        old_pos = np.array(skel_3d.get(b_name, [0, 0, 0]))
                        tri_pos = data['pos']
                        
                        # Translate to align with hand camera center
                        new_pos = tri_pos + translation
                        
                        # Validate reasonable distances from center (sanity check only)
                        dist_from_center = np.linalg.norm(new_pos - hand_center_world)
                        
                        # Hand should fit within ~12cm radius from center
                        max_dist = 0.12
                        
                        if dist_from_center > max_dist:
                            print(f"###   WARNING: {b_name} very far from center ({dist_from_center:.3f}m), clamping to {max_dist:.3f}m")
                            direction = (new_pos - hand_center_world) / (dist_from_center + 1e-8)
                            new_pos = hand_center_world + direction * max_dist
                        
                        # Debug for key joints
                        if '.1' in b_name or b_name == wrist_tri_key or 'Tip' in b_name:
                            movement = np.linalg.norm(new_pos - old_pos)
                            dist_from_center = np.linalg.norm(new_pos - hand_center_world)
                            dist_from_wrist = np.linalg.norm(new_pos - new_pos if b_name == wrist_tri_key else 
                                                             np.linalg.norm(new_pos - (triangulated_joints[wrist_tri_key]['pos'] + translation)))
                            print(f"###   {b_name}: moved {movement:.3f}m, {dist_from_center:.3f}m from center")
                        
                        skel_3d[b_name] = new_pos.tolist()
                        updated_count += 1
                
                print(f"### [AutoRig] Hand refinement complete: updated {updated_count} joints")
            else:
                print(f"### [AutoRig] WARNING: Hand {hand_short} not found in metadata")
        elif camera_metadata:
            # INITIAL TRIANGULATION PATH
            matrices = camera_metadata.get('matrices', [])
            num_frames = len(matrices)
            
            # Ensure dwpose_keypoints is a list and matches num_frames
            if isinstance(dwpose_keypoints, dict):
                dwpose_keypoints = [dwpose_keypoints] * num_frames
            elif isinstance(dwpose_keypoints, list):
                if len(dwpose_keypoints) != num_frames:
                    print(f"### [AutoRig] WARNING: dwpose_keypoints has {len(dwpose_keypoints)} entries but {num_frames} camera matrices")
                    print(f"### [AutoRig] Using available keypoints (may cause issues if mismatch is large)")
                    # Pad or truncate to match
                    if len(dwpose_keypoints) < num_frames:
                        dwpose_keypoints = dwpose_keypoints + [dwpose_keypoints[-1]] * (num_frames - len(dwpose_keypoints))
                    else:
                        dwpose_keypoints = dwpose_keypoints[:num_frames]
            
            print(f"### [AutoRig] Triangulating {num_frames} frames with RANSAC Outlier Rejection...")
            # Body Joints
            triangulated_keypoints = []
            face_keypoints = []
            for j_idx_str, b_name in OPENPOSE_MAP.items():
                j_idx = int(j_idx_str)
                joint_data = [extract_keypoint(dwpose_keypoints[i], j_idx) for i in range(num_frames)]
                # Check if this keypoint has valid data (not all None)
                if any(jd is not None for jd in joint_data):
                    skel_3d[b_name] = triangulate_ransac(joint_data, matrices)
                    triangulated_keypoints.append(b_name)
                    # Track face keypoints separately
                    if b_name.startswith("Face.") or b_name.startswith("Mouth."):
                        face_keypoints.append(b_name)
            
            print(f"### [AutoRig] Triangulated {len(triangulated_keypoints)} keypoints")
            if face_keypoints:
                print(f"### [AutoRig] Face keypoints detected: {', '.join(face_keypoints)}")
            else:
                print(f"### [AutoRig] No face keypoints detected (chin/mouth not available)")
            
            # Triangulate face keypoints from face_keypoints_2d array
            for face_idx_str, face_name in FACE_KEYPOINT_MAP.items():
                face_idx = int(face_idx_str)
                joint_data = [extract_face_keypoint(dwpose_keypoints[i], face_idx) for i in range(num_frames)]
                # Check if this keypoint has valid data
                if any(jd is not None for jd in joint_data):
                    skel_3d[face_name] = triangulate_ransac(joint_data, matrices)
                    face_keypoints.append(face_name)
            
            # Update face keypoint detection message
            if face_keypoints:
                print(f"### [AutoRig] Face keypoints triangulated: {', '.join(face_keypoints)}")

            # Hand Joints (Initial Pass)
            for hand_type in ["left", "right"]:
                suffix = ".L" if hand_type == "left" else ".R"
                for j_idx, b_base in HAND_JOINT_MAP.items():
                    b_name = b_base + suffix
                    joint_data = [extract_hand_keypoint(dwpose_keypoints[i], hand_type, j_idx) for i in range(num_frames)]
                    skel_3d[b_name] = triangulate_ransac(joint_data, matrices)

            # Hips (synthesize from thighs)
            if "Thigh.L" in skel_3d and "Thigh.R" in skel_3d:
                l_thigh, r_thigh = np.array(skel_3d["Thigh.L"]), np.array(skel_3d["Thigh.R"])
                if np.linalg.norm(l_thigh) > 0.001 and np.linalg.norm(r_thigh) > 0.001:
                    skel_3d["Hips"] = ((l_thigh + r_thigh) / 2).tolist()

            # Spine
            if "Hips" in skel_3d and "Neck" in skel_3d:
                h, n = np.array(skel_3d["Hips"]), np.array(skel_3d["Neck"])
                if np.linalg.norm(h) > 0.001:
                    skel_3d["Spine"] = ((h + n) / 2).tolist()

            # Head
            if "Eye.L" in skel_3d and "Eye.R" in skel_3d:
                l_eye, r_eye = np.array(skel_3d["Eye.L"]), np.array(skel_3d["Eye.R"])
                if np.linalg.norm(l_eye) > 0.001:
                    skel_3d["Head"] = ((l_eye + r_eye) / 2).tolist()
            elif "Ear.L" in skel_3d and "Ear.R" in skel_3d:
                l_ear, r_ear = np.array(skel_3d["Ear.L"]), np.array(skel_3d["Ear.R"])
                if np.linalg.norm(l_ear) > 0.001:
                    skel_3d["Head"] = ((l_ear + r_ear) / 2).tolist()

            # Apply joint kinks for better articulation (only after initial triangulation, not hand refinement)
            if not initial_skeleton:
                # Mirror better side if one side has significantly better detection
                mirror_better_side(skel_3d, dwpose_keypoints)
                
                print(f"### [AutoRig] Adding joint kinks for better articulation...")
                # Forward direction (toward front of body) is +Y in world space
                forward = [0, 0.05, 0]  # 5cm forward
                
                # Add kinks to elbows (Arm keypoints)
                add_joint_kink(skel_3d, "Arm.L", forward)
                add_joint_kink(skel_3d, "Arm.R", forward)
                
                # Add kinks to knees (Shin keypoints) 
                add_joint_kink(skel_3d, "Shin.L", forward)
                add_joint_kink(skel_3d, "Shin.R", forward)
                
                # Align spine/body on single plane in front view
                print(f"### [AutoRig] Aligning spine and body on center plane...")
                align_body_centerline(skel_3d)

        return (skel_3d,)

class UR_ModelSelector:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = Path(folder_paths.get_input_directory()) / "models"
        input_dir.mkdir(parents=True, exist_ok=True)
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.obj', '.fbx', '.glb', '.gltf'))]
        return {"required": {"model_file": (sorted(files) if files else ["No models found"],)}}
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("model_path", "preview")
    FUNCTION = "exec"
    CATEGORY = "UniversalRig"
    def exec(self, model_file):
        if "No models" in model_file: return ("", torch.zeros((1,512,512,3)))
        input_dir = Path(folder_paths.get_input_directory()) / "models"
        m_path = input_dir / model_file
        out_p = TEMP_DIR / f"pre_{model_file}.png"
        exe, _ = AutoRigInstaller.ensure_blender(NODE_ROOT)
        if not out_p.exists():
            cfg = {"mode": "preview", "model_path": str(m_path.absolute()), "output_path": str(out_p.absolute())}
            cfg_p = TEMP_DIR / "prev_cfg.json"
            with open(cfg_p, 'w') as f: json.dump(cfg, f)
            subprocess.run([str(exe), "-b", "--factory-startup", "-noaudio", "-P", str(NODE_ROOT / "blender_engine.py"), "--", str(cfg_p)], check=True)
        img = Image.open(out_p).convert("RGB")
        return (str(m_path), torch.from_numpy(np.array(img).astype(np.float32)/255.0)[None,])

class UR_RenderViews:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_path": ("STRING", {"forceInput": True}), "resolution": ("INT", {"default": 1024})}}
    RETURN_TYPES = ("IMAGE", "METADATA")
    FUNCTION = "exec"
    CATEGORY = "UniversalRig"
    def exec(self, model_path, resolution):
        out_dir = TEMP_DIR / "views"
        out_dir.mkdir(exist_ok=True)
        meta_p = TEMP_DIR / "cam_meta.json"
        exe, _ = AutoRigInstaller.ensure_blender(NODE_ROOT)
        cfg = {"mode": "render_views", "model_path": model_path, "output_dir": str(out_dir.absolute()), "meta_path": str(meta_p.absolute()), "res": resolution}
        cfg_p = TEMP_DIR / "render_cfg.json"
        with open(cfg_p, 'w') as f: json.dump(cfg, f)
        subprocess.run([str(exe), "-b", "--factory-startup", "-noaudio", "-P", str(NODE_ROOT / "blender_engine.py"), "--", str(cfg_p)], check=True)
        
        # Dynamically load all rendered views (detects actual file count)
        imgs = []
        i = 0
        while True:
            p = out_dir / f"v_{i}.png"
            if not p.exists():
                break
            imgs.append(torch.from_numpy(np.array(Image.open(p).convert("RGB")).astype(np.float32)/255.0))
            i += 1
        
        print(f"### [AutoRig] Loaded {len(imgs)} body view renders")
        with open(meta_p, 'r') as f: matrices = json.load(f)
        return (torch.stack(imgs), {"matrices": matrices, "model_path": model_path})

class UR_Finalize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"forceInput": True}),
                "skeleton_3d": ("SKELETON_3D",),
                "universal_template": ("STRING", {"default": str(UNIVERSAL_RIG_TEMPLATE)}),
            },
        }
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "exec"
    CATEGORY = "UniversalRig"

    def exec(self, model_path, skeleton_3d, universal_template=""):
        if not universal_template:
            universal_template = str(UNIVERSAL_RIG_TEMPLATE)
        output_dir = Path(folder_paths.get_output_directory())
        base_name = Path(model_path).stem
        suffix = "_universal_rigged"
        final_filename = f"{base_name}{suffix}.fbx"
        counter = 1
        while (output_dir / final_filename).exists():
            final_filename = f"{base_name}{suffix}_{counter}.fbx"
            counter += 1
        out_p = output_dir / final_filename
        meta_p = TEMP_DIR / "universal_meta.json"
        exe, _ = AutoRigInstaller.ensure_blender(NODE_ROOT)
        cfg = {
            "mode": "adapt_universal_rig",
            "model_path": model_path,
            "universal_template": universal_template,
            "skeleton": skeleton_3d,
            "export_path": str(out_p.absolute()),
            "meta_path": str(meta_p.absolute())
        }
        cfg_p = TEMP_DIR / "adapt_universal_cfg.json"
        with open(cfg_p, 'w') as f: json.dump(cfg, f)
        subprocess.run([str(exe), "-b", "--factory-startup", "-noaudio", "-P", str(NODE_ROOT / "blender_engine.py"), "--", str(cfg_p)], check=True)
        return (final_filename,)


class UR_MediaPipeHandDetector:
    """Alternative hand detection using MediaPipe for better finger accuracy."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "hand_type": (["left", "right"],),
                "min_detection_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    RETURN_TYPES = ("*", "IMAGE")
    RETURN_NAMES = ("dwpose_keypoints", "preview_images")
    FUNCTION = "exec"
    CATEGORY = "UniversalRig"
    
    def exec(self, images, hand_type, min_detection_confidence):
        """
        Process images using MediaPipe hand detection.
        
        Args:
            images: Batch of images (B, H, W, C) in 0-1 float range
            hand_type: 'left' or 'right'
            min_detection_confidence: Minimum confidence threshold
            
        Returns:
            tuple: (keypoints, preview_images)
        """
        try:
            from .mediapipe_hand_detector import MediaPipeHandDetector
            import cv2
            import mediapipe as mp
        except ImportError as e:
            error_msg = f"""
### [AutoRig] MediaPipe not available. 

Please install manually by running these commands in your terminal:

Windows:
    cd C:\\ComfyUI\\ComfyUI_windows_portable
    python_embeded\\python.exe -m pip install --upgrade pip
    python_embeded\\python.exe -m pip install mediapipe opencv-python

Linux/Mac:
    pip install mediapipe opencv-python

Then restart ComfyUI.

Error details: {str(e)}
"""
            print(error_msg)
            raise ImportError(error_msg)
        
        detector = MediaPipeHandDetector(
            static_image_mode=True,
            max_num_hands=1,  # Process one hand at a time
            min_detection_confidence=min_detection_confidence
        )
        
        results = []
        preview_images = []
        detected_count = 0
        
        # Save images temporarily for MediaPipe processing
        temp_images = []
        for i in range(images.shape[0]):
            img = images[i].cpu().numpy()
            img_uint8 = (img * 255).astype(np.uint8)
            temp_path = TEMP_DIR / f"mediapipe_temp_{i}.png"
            Image.fromarray(img_uint8).save(temp_path)
            temp_images.append(temp_path)
        
        # Process each image
        for i, temp_path in enumerate(temp_images):
            # Get detection
            detection = detector.detect_hands(temp_path)
            
            # Debug: Log detection results (removed for cleaner output)
            
            # Create preview image with landmarks drawn
            image_bgr = cv2.imread(str(temp_path))
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Draw landmarks on the image
            hand_key = f'hand_{hand_type}_keypoints_2d'
            detected_this_frame = False
            
            if detection['people'] and hand_key in detection['people'][0]:
                keypoints = detection['people'][0][hand_key]
                if len(keypoints) > 0:
                    detected_this_frame = True
                    detected_count += 1
                    
                    # Draw hand landmarks
                    for j in range(21):
                        idx = j * 3
                        if idx + 2 < len(keypoints):
                            x = int(keypoints[idx])
                            y = int(keypoints[idx + 1])
                            conf = keypoints[idx + 2]
                            
                            # Draw landmark point
                            if conf > 0.3:
                                cv2.circle(image_rgb, (x, y), 5, (0, 255, 0), -1)
                                cv2.circle(image_rgb, (x, y), 6, (255, 255, 255), 1)
                    
                    # Draw connections
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                        (5, 9), (9, 13), (13, 17)  # Palm
                    ]
                    
                    for conn in connections:
                        start_idx = conn[0] * 3
                        end_idx = conn[1] * 3
                        if start_idx + 2 < len(keypoints) and end_idx + 2 < len(keypoints):
                            x1, y1, conf1 = int(keypoints[start_idx]), int(keypoints[start_idx + 1]), keypoints[start_idx + 2]
                            x2, y2, conf2 = int(keypoints[end_idx]), int(keypoints[end_idx + 1]), keypoints[end_idx + 2]
                            if conf1 > 0.3 and conf2 > 0.3:
                                cv2.line(image_rgb, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    
                    cv2.putText(image_rgb, f"{hand_type.upper()} HAND DETECTED", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if not detected_this_frame:
                cv2.putText(image_rgb, "NO HAND DETECTED", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Convert preview to tensor format
            preview_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
            preview_images.append(preview_tensor)
            
            results.append(detection)
            
            # Clean up temp file
            temp_path.unlink()
        
        detector.close()
        

        
        # Stack preview images into batch
        preview_batch = torch.stack(preview_images)
        
        return (results, preview_batch)

class UR_RefineHandViews:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"forceInput": True}),
                "skeleton_3d": ("SKELETON_3D",),
                "resolution": ("INT", {"default": 512}),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "METADATA")
    RETURN_NAMES = ("left_hand_images", "right_hand_images", "hand_cam_metadata")
    FUNCTION = "exec"
    CATEGORY = "UniversalRig"

    def exec(self, model_path, skeleton_3d, resolution):
        out_dir = TEMP_DIR / "hand_views"
        out_dir.mkdir(exist_ok=True)
        meta_p = TEMP_DIR / "hand_cam_meta.json"
        
        # Debug: Print skeleton keys to see what joints we have
        print(f"### [AutoRig_RefineHandViews] Skeleton has {len(skeleton_3d)} joints:")
        hand_joints = [k for k in skeleton_3d.keys() if any(x in k for x in ["Hand", "Thumb", "Index", "Middle", "Ring", "Pinky"])]
        print(f"### [AutoRig_RefineHandViews] Hand-related joints: {hand_joints}")
        
        exe, _ = AutoRigInstaller.ensure_blender(NODE_ROOT)
        cfg = {
            "mode": "render_hand_views",
            "model_path": model_path,
            "skeleton": skeleton_3d,
            "output_dir": str(out_dir.absolute()),
            "meta_path": str(meta_p.absolute()),
            "res": resolution
        }
        cfg_p = TEMP_DIR / "render_hands_cfg.json"
        with open(cfg_p, 'w') as f:
            json.dump(cfg, f)
        
        # Run Blender and capture output
        print(f"### [AutoRig_RefineHandViews] Running Blender to render hand views...")
        result = subprocess.run(
            [str(exe), "-b", "--factory-startup", "-noaudio", "-P", str(NODE_ROOT / "blender_engine.py"), "--", str(cfg_p)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print Blender's output
        if result.stdout:
            print("### [Blender stdout]:")
            print(result.stdout)
        if result.stderr:
            print("### [Blender stderr]:")
            print(result.stderr)
        
        with open(meta_p, 'r') as f:
            hand_meta = json.load(f)
        
        print(f"### [AutoRig_RefineHandViews] Metadata keys: {list(hand_meta.keys())}")

        def load_imgs(hand_suffix):
            images = []
            if hand_suffix in hand_meta:
                print(f"### [AutoRig_RefineHandViews] Loading {len(hand_meta[hand_suffix]['image_paths'])} images for hand {hand_suffix}")
                for p_str in hand_meta[hand_suffix]["image_paths"]:
                    p = Path(p_str)
                    if p.exists():
                        images.append(torch.from_numpy(np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0))
                    else:
                        print(f"### [AutoRig_RefineHandViews] Image not found: {p}")
            else:
                print(f"### [AutoRig_RefineHandViews] Hand {hand_suffix} not found in metadata")
            
            if not images:
                print(f"### [AutoRig_RefineHandViews] No images loaded for hand {hand_suffix}, returning black image")
                return torch.zeros((1, resolution, resolution, 3), dtype=torch.float32)
            
            return torch.stack(images)

        l_imgs = load_imgs("L")
        r_imgs = load_imgs("R")

        return (l_imgs, r_imgs, hand_meta)

NODE_CLASS_MAPPINGS = {
    "UR_ModelSelector": UR_ModelSelector,
    "UR_RenderViews": UR_RenderViews,
    "UR_Triangulate": UR_Triangulate,
    "UR_RefineHandViews": UR_RefineHandViews,
    "UR_MediaPipeHandDetector": UR_MediaPipeHandDetector,
    "UR_Finalize": UR_Finalize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UR_ModelSelector": "UniversalRig: Model Selector",
    "UR_RenderViews": "UniversalRig: Render Views",
    "UR_Triangulate": "UniversalRig: Triangulate 3D",
    "UR_RefineHandViews": "UniversalRig: Refine Hand Views",
    "UR_MediaPipeHandDetector": "UniversalRig: MediaPipe Hands",
    "UR_Finalize": "UniversalRig: Finalize (AI2Mesh)",
}