from .anchor_generator import (
    AnchorGeneratorRange,
    AnchorGeneratorStride,
    BevAnchorGeneratorRange,
)
from .target_assigner import TargetAssigner
from .target_ops_v2 import create_target_np

# target_ops: default
# target_ops_v2: for default cia-ssd
# target_ops_v3: for anchors_mask, but inconsistent with outside code process
# target_ops_v4: for anchors_mask, and consistent with outside code process
