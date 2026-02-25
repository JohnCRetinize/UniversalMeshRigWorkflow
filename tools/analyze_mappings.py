import json
from pathlib import Path

# Simple analysis to evaluate which keypoint mapping best matches
# your DWPoser/OpenPose JSON by checking left/right symmetry and
# average confidences per joint.

THR_CONF = 0.2

SYMMETRIC_PAIRS = [
    ("Shoulder.L", "Shoulder.R"), ("Arm.L", "Arm.R"), ("Forearm.L", "Forearm.R"),
    ("Hand.L", "Hand.R"), ("Thigh.L", "Thigh.R"), ("Shin.L", "Shin.R"),
    ("Foot.L", "Foot.R"), ("Eye.L", "Eye.R"), ("Ear.L", "Ear.R")
]

# Candidate mappings (copied from your __init__.py)
COCO17_MAP = {
    "0": "Nose",
    "5": "Shoulder.L", "7": "Arm.L", "9": "Forearm.L",
    "6": "Shoulder.R", "8": "Arm.R", "10": "Forearm.R",
    "11": "Thigh.L", "13": "Shin.L", "15": "Foot.L",
    "12": "Thigh.R", "14": "Shin.R", "16": "Foot.R",
    "1": "Eye.L", "2": "Eye.R",
    "3": "Ear.L", "4": "Ear.R"
}

COCO18_MAP = {
    "0": "Nose", "1": "Neck",
    "2": "Shoulder.R", "3": "Arm.R", "4": "Forearm.R",
    "5": "Shoulder.L", "6": "Arm.L", "7": "Forearm.L",
    "8": "Thigh.R", "9": "Shin.R", "10": "Foot.R",
    "11": "Thigh.L", "12": "Shin.L", "13": "Foot.L",
    "14": "Eye.R", "15": "Eye.L",
    "16": "Ear.R", "17": "Ear.L"
}

BODY25_MAP = {
    "0": "Nose", "1": "Neck",
    "2": "Shoulder.R", "3": "Arm.R", "4": "Forearm.R",
    "5": "Shoulder.L", "6": "Arm.L", "7": "Forearm.L",
    "8": "Hips",
    "9": "Thigh.R", "10": "Shin.R", "11": "Foot.R",
    "12": "Thigh.L", "13": "Shin.L", "14": "Foot.L",
    "15": "Eye.R", "16": "Eye.L",
    "17": "Ear.R", "18": "Ear.L",
}

MEDIAPIPE_MAP = {
    "0": "Nose",
    "11": "Shoulder.L", "13": "Arm.L", "15": "Forearm.L",
    "12": "Shoulder.R", "14": "Arm.R", "16": "Forearm.R",
    "23": "Thigh.L", "25": "Shin.L", "27": "Foot.L",
    "24": "Thigh.R", "26": "Shin.R", "28": "Foot.R",
    "2": "Eye.L", "5": "Eye.R",
    "7": "Ear.L", "8": "Ear.R"
}

HALPE26_MAP = {
    "0": "Nose",
    "5": "Shoulder.L", "7": "Arm.L", "9": "Forearm.L",
    "6": "Shoulder.R", "8": "Arm.R", "10": "Forearm.R",
    "11": "Thigh.L", "13": "Shin.L", "15": "Foot.L",
    "12": "Thigh.R", "14": "Shin.R", "16": "Foot.R",
    "1": "Eye.L", "2": "Eye.R",
    "3": "Ear.L", "4": "Ear.R",
    "17": "Head", "18": "Neck", "19": "Hips"
}

MAPPINGS = {
    "COCO17": COCO17_MAP,
    "COCO18": COCO18_MAP,
    "BODY25": BODY25_MAP,
    "MEDIAPIPE": MEDIAPIPE_MAP,
    "HALPE26": HALPE26_MAP,
}


def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze(path):
    data = load_json(path)

    results = {}
    for name, mapping in MAPPINGS.items():
        rev = {v: int(k) for k, v in mapping.items()}
        violations = 0
        checks = 0
        conf_sums = {}
        conf_counts = {}

        for frame in data:
            people = frame.get('people') or []
            if not people: continue
            p0 = people[0]
            kps = p0.get('pose_keypoints_2d') or []
            canvas_w = frame.get('canvas_width', 1024)
            # build index->(x,y,c)
            n = len(kps) // 3
            pts = {}
            for i in range(n):
                x = kps[3*i]
                y = kps[3*i+1]
                c = kps[3*i+2]
                pts[i] = (x, y, c)
                conf_sums.setdefault(i, 0.0)
                conf_counts.setdefault(i, 0)
                conf_sums[i] += c
                conf_counts[i] += 1

            for L, R in SYMMETRIC_PAIRS:
                if L in rev and R in rev:
                    li = rev[L]
                    ri = rev[R]
                    if li in pts and ri in pts:
                        lx, ly, lc = pts[li]
                        rx, ry, rc = pts[ri]
                        if lc >= THR_CONF and rc >= THR_CONF:
                            checks += 1
                            # image origin is top-left: left-side has smaller x
                            if lx > rx:
                                violations += 1

        avg_conf = {i: (conf_sums[i]/conf_counts[i] if conf_counts[i] else 0.0) for i in conf_sums}
        results[name] = {
            'checks': checks,
            'violations': violations,
            'violation_rate': (violations / checks) if checks else None,
            'avg_conf_per_index': avg_conf,
            'mapping_size': len(mapping)
        }

    return results


def pretty_print(results):
    sorted_results = sorted(results.items(), key=lambda x: (1e9 if x[1]['violation_rate'] is None else x[1]['violation_rate']))
    for name, r in sorted_results:
        print(f"Mapping: {name}")
        print(f"  mapping_size: {r['mapping_size']}")
        print(f"  checks: {r['checks']}")
        print(f"  violations: {r['violations']}")
        print(f"  violation_rate: {r['violation_rate']}")
        # show a few low-confidence indices
        low = sorted(r['avg_conf_per_index'].items(), key=lambda x: x[1])[:5]
        print(f"  lowest avg conf indices: {low}")
        print()


if __name__ == '__main__':
    base = Path(__file__).resolve().parents[1]
    jpath = base / 'output' / 'DEBUG_keypoints.json'
    if not jpath.exists():
        print('DEBUG_keypoints.json not found at', jpath)
        print('Place the DWPoser output at that path or edit this script to point to it.')
    else:
        res = analyze(jpath)
        pretty_print(res)
