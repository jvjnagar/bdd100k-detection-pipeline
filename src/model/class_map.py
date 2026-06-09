"""BDD100K class-ID mapping constants.

Single source of truth for the 10 detection class names and their integer IDs.
Used by the DataLoader (for label tensors) and the evaluation scripts
(for decoding predictions back to human-readable names).
"""

# BDD100K 10 detection classes mapped to contiguous integer IDs (0-indexed).
CLASS_TO_ID = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "train": 3,
    "person": 4,
    "rider": 5,
    "bike": 6,
    "motor": 7,
    "traffic light": 8,
    "traffic sign": 9,
}

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
