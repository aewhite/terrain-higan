"""
Shared constants for Zn upscaler and patch generator training.
Used by train_zn_generator and train_zn_patch_generator.
"""
from typing import List, Tuple

PATCH_SIZES: List[int] = [
    3, 4, 4, 6, 6, 8, 8, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
]

OUTPUT_RANGES: List[Tuple[float, float]] = [
    (-6.1464, 5.0754),
    (-6.4154, 5.2886),
    (-7.3069, 5.4440),
    (-8.4980, 5.7134),
    (-9.5342, 6.1157),
    (-10.2123, 6.2814),
    (-10.5293, 6.9329),
    (-10.7086, 7.5437),
    (-10.7519, 8.0639),
    (-10.7679, 8.3190),
    (-10.7812, 8.5060),
]

DIFF_RANGES: List[Tuple[float, float]] = [
    (0.0, 0.0),
    (-5.0171, 4.7593),
    (-4.1702, 3.9804),
    (-4.1363, 3.9781),
    (-3.9231, 3.6457),
    (-3.1264, 3.2273),
    (-2.4727, 2.8321),
    (-2.4284, 2.6252),
    (-3.1311, 2.1894),
    (-2.5987, 2.6701),
    (-3.8003, 2.9217),
]

# log(std) means per z; patch_std = exp(patch_log_std_means[z])
PATCH_LOG_STD_MEANS: List[float] = [
    -0.4800,
    -0.6345,
    -1.1257,
    -1.2365,
    -1.6962,
    -1.8644,
    -2.2278,
    -2.4588,
    -2.8588,
    -3.3685,
    -3.9188,
]

PATCH_LOG_STD_STDS: List[float] = [
     0.7999,
     0.8602,
     0.9823,
     1.0061,
     1.0884,
     1.1272,
     1.2001,
     1.2563,
     1.3223,
     1.3905,
     1.4594,
]

# Legacy per-zoom kilometer statistics retained for offline analysis only.
PATCH_MEAN_MEANS: List[float] = [
    -2.4589,
    -2.4595,
    -2.4565,
    -2.4514,
    -2.4379,
    -2.4131,
    -2.3584,
    -2.3924,
    -2.3721,
    -2.3523,
    -2.3540,
]

PATCH_MEAN_STDS: List[float] = [
    2.0468,
    2.1521,
    2.2771,
    2.3110,
    2.3591,
    2.3691,
    2.3948,
    2.3919,
    2.4196,
    2.4101,
    2.4333,
]

# ========================================================
# Zoom Diff Statistics (avg_pool2d -> nearest upsample)
# ========================================================

PATCH_DIFF_LOG_STD_MEANS: List[float] = [
    0.0,
    -0.58,
    -0.89,
    -1.20,
    -1.50,
    -1.77,
    -2.08,
    -2.47,
    -2.92,
    -3.43,
    -3.95,
]
