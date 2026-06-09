from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional

@dataclass(frozen=True)
class FrameInput:
    frame_id: str
    image_path: str
    yolo_path: str
    pose_json_path: Optional[str]

def iter_dataset_frames(dataset_root: str, max_frames: int=0) -> List[FrameInput]:
    img_dir = os.path.join(dataset_root, 'images')
    yolo_dir = os.path.join(dataset_root, 'labels_yolo')
    pose_dir = os.path.join(dataset_root, 'labels_pose')
    if not os.path.isdir(img_dir):
        return []
    names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith('.png'))
    if max_frames > 0:
        names = names[:max_frames]
    out: List[FrameInput] = []
    for name in names:
        stem = os.path.splitext(name)[0]
        yolo_p = os.path.join(yolo_dir, f'{stem}.txt')
        pose_p = os.path.join(pose_dir, f'{stem}.json')
        out.append(FrameInput(frame_id=stem, image_path=os.path.join(img_dir, name), yolo_path=yolo_p if os.path.isfile(yolo_p) else '', pose_json_path=pose_p if os.path.isfile(pose_p) else None))
    return out

def iter_frames(frames: List[FrameInput]) -> Iterator[FrameInput]:
    yield from frames
