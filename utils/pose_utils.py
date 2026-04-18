"""
Pose and rotation helper utilities for camera interpolation workflows.
"""

from __future__ import annotations

import math

import numpy as np


class CameraPoseInterpolator:
    """Pose matching and incremental interpolation helper used by local Difix3D trainers."""

    def __init__(self, rotation_weight: float = 1.0, translation_weight: float = 1.0) -> None:
        self.rotation_weight = float(rotation_weight)
        self.translation_weight = float(translation_weight)

    def compute_pose_distance(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        t1 = np.asarray(pose1[:3, 3], dtype=np.float64)
        t2 = np.asarray(pose2[:3, 3], dtype=np.float64)
        translation_dist = float(np.linalg.norm(t1 - t2))

        q1 = rotation_matrix_to_quaternion(pose1[:3, :3])
        q2 = rotation_matrix_to_quaternion(pose2[:3, :3])
        if float(np.dot(q1, q2)) < 0.0:
            q2 = -q2
        dot = float(np.clip(np.dot(q1, q2), -1.0, 1.0))
        rotation_dist = math.acos(float(np.clip(2.0 * dot * dot - 1.0, -1.0, 1.0)))
        return self.translation_weight * translation_dist + self.rotation_weight * rotation_dist

    def find_nearest_assignments(self, training_poses: np.ndarray, testing_poses: np.ndarray) -> list[int]:
        assignments: list[int] = []
        for test_pose in testing_poses:
            distances = [self.compute_pose_distance(train_pose, test_pose) for train_pose in training_poses]
            assignments.append(int(np.argmin(distances)))
        return assignments

    def interpolate_rotation(self, rot_a: np.ndarray, rot_b: np.ndarray, t: float) -> np.ndarray:
        qa = rotation_matrix_to_quaternion(rot_a)
        qb = rotation_matrix_to_quaternion(rot_b)
        q = slerp_quaternion(qa, qb, float(np.clip(t, 0.0, 1.0)))
        return quaternion_to_rotation_matrix(q)

    def interpolate_poses(
        self,
        training_poses: np.ndarray,
        testing_poses: np.ndarray,
        num_steps: int = 20,
    ) -> list[list[np.ndarray]]:
        assignments = self.find_nearest_assignments(training_poses, testing_poses)
        sequences: list[list[np.ndarray]] = []
        for test_idx, train_idx in enumerate(assignments):
            train_pose = training_poses[train_idx]
            test_pose = testing_poses[test_idx]
            seq: list[np.ndarray] = []
            for t in np.linspace(0.0, 1.0, num_steps):
                pose_interp = interpolate_pose(train_pose, test_pose, float(t))
                seq.append(pose_interp)
            sequences.append(seq)
        return sequences

    def shift_poses(
        self,
        training_poses: np.ndarray,
        testing_poses: np.ndarray,
        distance: float = 0.1,
        threshold: float = 0.1,
    ) -> np.ndarray:
        del threshold
        assignments = self.find_nearest_assignments(training_poses, testing_poses)
        shifted: list[np.ndarray] = []
        for test_idx, train_idx in enumerate(assignments):
            train_pose = np.asarray(training_poses[train_idx], dtype=np.float64)
            test_pose = np.asarray(testing_poses[test_idx], dtype=np.float64)

            if self.compute_pose_distance(train_pose, test_pose) <= distance:
                shifted.append(test_pose.astype(np.float32))
                continue

            t1 = train_pose[:3, 3]
            t2 = test_pose[:3, 3]
            translation_direction = t2 - t1
            translation_norm = float(np.linalg.norm(translation_direction))

            if translation_norm > 1e-6:
                translation_step = (translation_direction / translation_norm) * distance
                new_translation = t1 + translation_step
            else:
                new_translation = t2

            if np.dot(new_translation - t1, t2 - t1) <= 0.0 or np.linalg.norm(new_translation - t2) <= distance:
                new_translation = t2

            if translation_norm > 1e-6:
                interp_t = min(distance / translation_norm, 1.0)
                rot_interp = self.interpolate_rotation(train_pose[:3, :3], test_pose[:3, :3], interp_t)
            else:
                rot_interp = np.asarray(test_pose[:3, :3], dtype=np.float64)

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rot_interp.astype(np.float32)
            pose[:3, 3] = new_translation.astype(np.float32)
            shifted.append(pose)
        return np.stack(shifted, axis=0).astype(np.float32)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 1e-12)) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 1e-12)) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 1e-12)) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    return q


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def slerp_quaternion(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-12)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / max(float(np.linalg.norm(q)), 1e-12)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.sin(theta_0 - theta) / max(sin_theta_0, 1e-12)
    s1 = sin_theta / max(sin_theta_0, 1e-12)
    q = s0 * q0 + s1 * q1
    return q / max(float(np.linalg.norm(q)), 1e-12)


def interpolate_pose(c2w_a: np.ndarray, c2w_b: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 0.0:
        return c2w_a.astype(np.float32).copy()
    if t >= 1.0:
        return c2w_b.astype(np.float32).copy()
    qa = rotation_matrix_to_quaternion(c2w_a[:3, :3])
    qb = rotation_matrix_to_quaternion(c2w_b[:3, :3])
    q = slerp_quaternion(qa, qb, t)
    r = quaternion_to_rotation_matrix(q)
    trans = (1.0 - t) * c2w_a[:3, 3] + t * c2w_b[:3, 3]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = r.astype(np.float32)
    out[:3, 3] = trans.astype(np.float32)
    return out
