from __future__ import annotations

import cv2
import torch
from PIL import Image

from datasets.utkface import build_transforms
from utils.checkpoint import load_age_model
from utils.calibration import AgeCalibrator
from utils.metrics import forward_with_tta, outputs_to_age


class AgePredictor:
    def __init__(
        self,
        checkpoint_path,
        device: str | None = None,
        backbone: str | None = None,
        tta: bool = False,
        calibration: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.checkpoint = load_age_model(checkpoint_path, self.device, backbone)
        self.method = self.checkpoint.get("method", "dldl")
        self.tta = tta
        self.calibrator = AgeCalibrator.from_json(calibration) if calibration else None
        self.transform = build_transforms(
            train=False,
            image_size=self.checkpoint.get("image_size", 224),
        )

    @torch.inference_mode()
    def predict_bgr_faces(self, frame, boxes) -> list[float]:
        tensors = []
        for x1, y1, x2, y2 in boxes:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(Image.fromarray(rgb)))
        if not tensors:
            return []
        batch = torch.stack(tensors).to(self.device)
        outputs = forward_with_tta(self.model, batch, self.method, enabled=self.tta)
        ages = outputs_to_age(outputs, self.method)
        if self.calibrator is not None:
            ages = self.calibrator.apply_tensor(ages)
        return ages.cpu().tolist()
