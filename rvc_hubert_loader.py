from __future__ import annotations

"""RVC-compatible loader for the exported SensorHubert model.

Copy this file into your RVC directory alongside sensor_hubert_rvc.pth.
Then replace RVC's HuBERT loading code with load_sensor_hubert().

RVC v2 normally calls:
    feats, _ = model.extract_features(source, padding_mask=mask, output_layer=12)
RVCHubert below implements exactly that interface on top of transformers HubertModel.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel, HubertConfig

META_NAME = "sensor_hubert_meta.json"


def normalize_source(x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Per-utterance zero-mean/unit-var (= fairseq F.layer_norm(wav, wav.shape), eps 1e-5),
    statistics over non-padded samples only. Idempotent up to eps, so harmless if the
    RVC caller already normalised."""
    x = x.float()
    if padding_mask is None or padding_mask.shape != x.shape:
        return F.layer_norm(x, x.shape[-1:])
    valid = (~padding_mask).float()
    n = valid.sum(-1, keepdim=True).clamp(min=1.0)
    m = (x * valid).sum(-1, keepdim=True) / n
    v = (((x - m) * valid) ** 2).sum(-1, keepdim=True) / n
    return (x - m) / torch.sqrt(v + 1e-5) * valid


class RVCHubert(nn.Module):
    """Wraps a transformers HubertModel with fairseq-style extract_features().

    normalize_input=True for students trained against a task.normalize=True teacher
    (mHuBERT-147): the waveform is normalised here, so RVC's own pipeline (which does
    not normalise at inference) cannot feed the student a mis-scaled input."""

    def __init__(self, hubert: HubertModel, normalize_input: bool = False) -> None:
        super().__init__()
        self.model = hubert
        self.normalize_input = normalize_input

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        output_layer: int = 12,
    ) -> tuple[torch.Tensor, None]:
        # fairseq padding_mask: True = padded; transformers attention_mask: 1 = real
        if self.normalize_input:
            source = normalize_source(source, padding_mask)
        attention_mask = None
        if padding_mask is not None:
            attention_mask = (~padding_mask).long()

        out = self.model(
            source,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # hidden_states[0] = CNN output; hidden_states[1..12] = transformer layers
        feats = out.hidden_states[output_layer]  # [B, T, 768]
        return feats, None

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def load_sensor_hubert(
    pth_path: str,
    config_path: str,
    device: str = "cpu",
    normalize_input: bool | None = None,
) -> RVCHubert:
    """Load the exported SensorHubert for RVC inference.

    Args:
        pth_path        : path to sensor_hubert_rvc.pth (state dict)
        config_path     : path to the HuBERT config.json (inside hf_model/)
        device          : "cpu" or "cuda"
        normalize_input : None → read sensor_hubert_meta.json next to the .pth
                          (exports without it = old raw-audio pipeline = False)
    """
    if normalize_input is None:
        meta = Path(pth_path).with_name(META_NAME)
        normalize_input = bool(json.loads(meta.read_text())["normalize_input"]) if meta.exists() else False
    config = HubertConfig.from_pretrained(config_path)
    base = HubertModel(config)
    state_dict = torch.load(pth_path, map_location=device)
    base.load_state_dict(state_dict)
    base.eval()
    model = RVCHubert(base, normalize_input=normalize_input)
    model.eval()
    return model.to(device)
