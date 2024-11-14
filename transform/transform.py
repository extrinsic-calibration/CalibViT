import torch
import numpy as np 
from scipy.spatial.transform import Rotation
from typing import Tuple, List, Optional, Union, Any
from pytorch3d import transforms

def get_transformation_matrix(rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Generate a transformation matrix from rotation and translation tensors."""
    rot_matrix = transforms.euler_angles_to_matrix(rot, 'XYZ') 
    tf = get_transform_from_rotation_translation_tensor(rotation=rot_matrix, translation=trans)
    return tf

def transform_point_cloud(pts: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Transform a point cloud using a transformation matrix."""
    R = trans[:, :3, :3]
    T = trans[:, :3, 3]
    pts = pts @ R.mT + T
    return pts

def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Get rigid transform matrix from rotation matrix and translation vector."""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform

def get_transform_from_rotation_translation_tensor(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Get rigid transform matrix from rotation matrix and translation vector."""
    tf = torch.zeros([rotation.shape[0], 4, 4], device=rotation.device)
    tf[:, :3, :3] = rotation
    tf[:, :3, 3] = translation
    tf[:, 3, 3] = 1
    return tf

def get_rotation_translation_from_transform(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract rotation matrix and translation vector from a rigid transform matrix."""
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation

def quaternion_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert a rotation matrix to a quaternion."""
    if matrix.shape[-2:] == (4, 4):
        R = matrix[:, :-1, :-1]
    elif matrix.shape[-2:] == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros((R.size(0), 4), device=matrix.device)

    cond1 = tr > 0
    S = torch.where(cond1, (tr + 1.0).sqrt() * 2, torch.zeros_like(tr))
    q[cond1, 0] = 0.25 * S[cond1]
    q[cond1, 1] = (R[cond1, 2, 1] - R[cond1, 1, 2]) / S[cond1]
    q[cond1, 2] = (R[cond1, 0, 2] - R[cond1, 2, 0]) / S[cond1]
    q[cond1, 3] = (R[cond1, 1, 0] - R[cond1, 0, 1]) / S[cond1]

    cond2 = ~cond1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    S = torch.where(cond2, (1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]).sqrt() * 2, S)
    q[cond2, 0] = (R[cond2, 2, 1] - R[cond2, 1, 2]) / S[cond2]
    q[cond2, 1] = 0.25 * S[cond2]
    q[cond2, 2] = (R[cond2, 0, 1] + R[cond2, 1, 0]) / S[cond2]
    q[cond2, 3] = (R[cond2, 0, 2] + R[cond2, 2, 0]) / S[cond2]

    cond3 = ~cond1 & ~cond2 & (R[:, 1, 1] > R[:, 2, 2])
    S = torch.where(cond3, (1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]).sqrt() * 2, S)
    q[cond3, 0] = (R[cond3, 0, 2] - R[cond3, 2, 0]) / S[cond3]
    q[cond3, 1] = (R[cond3, 0, 1] + R[cond3, 1, 0]) / S[cond3]
    q[cond3, 2] = 0.25 * S[cond3]
    q[cond3, 3] = (R[cond3, 1, 2] + R[cond3, 2, 1]) / S[cond3]

    cond4 = ~cond1 & ~cond2 & ~cond3
    S = torch.where(cond4, (1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]).sqrt() * 2, S)
    q[cond4, 0] = (R[cond4, 1, 0] - R[cond4, 0, 1]) / S[cond4]
    q[cond4, 1] = (R[cond4, 0, 2] + R[cond4, 2, 0]) / S[cond4]
    q[cond4, 2] = (R[cond4, 1, 2] + R[cond4, 2, 1]) / S[cond4]
    q[cond4, 3] = 0.25 * S[cond4]

    return q / q.norm(dim=-1, keepdim=True)

def quat2mat(q: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion to a rotation matrix."""
    assert q.shape[-1] == 4, "Not a valid quaternion"
    q = q / q.norm(dim=-1, keepdim=True)
    
    B = q.size(0)
    mat = torch.zeros((B, 4, 4), device=q.device)

    mat[:, 0, 0] = 1 - 2 * q[:, 2]**2 - 2 * q[:, 3]**2
    mat[:, 0, 1] = 2 * q[:, 1] * q[:, 2] - 2 * q[:, 3] * q[:, 0]
    mat[:, 0, 2] = 2 * q[:, 1] * q[:, 3] + 2 * q[:, 2] * q[:, 0]
    mat[:, 1, 0] = 2 * q[:, 1] * q[:, 2] + 2 * q[:, 3] * q[:, 0]
    mat[:, 1, 1] = 1 - 2 * q[:, 1]**2 - 2 * q[:, 3]**2
    mat[:, 1, 2] = 2 * q[:, 2] * q[:, 3] - 2 * q[:, 1] * q[:, 0]
    mat[:, 2, 0] = 2 * q[:, 1] * q[:, 3] - 2 * q[:, 2] * q[:, 0]
    mat[:, 2, 1] = 2 * q[:, 2] * q[:, 3] + 2 * q[:, 1] * q[:, 0]
    mat[:, 2, 2] = 1 - 2 * q[:, 1]**2 - 2 * q[:, 2]**2
    mat[:, 3, 3] = 1.0
    
    return mat

def tvector2mat(t: torch.Tensor) -> torch.Tensor:
    """Convert translation vectors to homogeneous transformation matrices."""
    assert t.shape[-1] == 3, "Not a valid translation"
    
    B = t.size(0)
    mat = torch.eye(4, device=t.device).unsqueeze(0).repeat(B, 1, 1)
    mat[:, 0, 3] = t[:, 0]
    mat[:, 1, 3] = t[:, 1]
    mat[:, 2, 3] = t[:, 2]
    
    return mat

def mat2xyzrpy(rotmatrix: torch.Tensor) -> torch.Tensor:
    """Decompose transformation matrices into components (XYZ and Roll-Pitch-Yaw)."""
    B = rotmatrix.size(0)
    x = rotmatrix[:, 0, 3]
    y = rotmatrix[:, 1, 3]
    z = rotmatrix[:, 2, 3]
    
    roll = torch.atan2(-rotmatrix[:, 1, 2], rotmatrix[:, 2, 2])
    pitch = torch.atan2(rotmatrix[:, 0, 2], torch.sqrt(rotmatrix[:, 1, 2]**2 + rotmatrix[:, 2, 2]**2))
    yaw = torch.atan2(-rotmatrix[:, 0, 1], rotmatrix[:, 0, 0])
    
    return torch.stack([x, y, z, roll, pitch, yaw], dim=-1)