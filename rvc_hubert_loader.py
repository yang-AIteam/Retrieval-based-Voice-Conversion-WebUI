from __future__ import annotations

"""RVC-compatible loader for the exported SensorHubert model.

Copy this file into your RVC directory alongside sensor_hubert_rvc.pth.
Then replace RVC's HuBERT loading code with load_sensor_hubert().

RVC v2 normally calls:
    feats, _ = model.extract_features(source, padding_mask=mask, output_layer=12)
RVCHubert below implements exactly that interface on top of transformers HubertModel.
"""

import torch
import torch.nn as nn
from transformers import HubertModel, HubertConfig


class RVCHubert(nn.Module):
    """Wraps a transformers HubertModel with fairseq-style extract_features()."""

    def __init__(self, hubert: HubertModel) -> None:
        super().__init__()
        self.model = hubert

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        output_layer: int = 12,
    ) -> tuple[torch.Tensor, None]:
        # fairseq padding_mask: True = padded; transformers attention_mask: 1 = real
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
) -> RVCHubert:
    """Load the exported SensorHubert for RVC inference.

    Args:
        pth_path    : path to sensor_hubert_rvc.pth (state dict)
        config_path : path to the HuBERT config.json (inside hf_model/)
        device      : "cpu" or "cuda"
    """
    config = HubertConfig.from_pretrained(config_path)
    base = HubertModel(config)
    state_dict = torch.load(pth_path, map_location=device)
    base.load_state_dict(state_dict)
    base.eval()
    model = RVCHubert(base)
    model.eval()
    return model.to(device)
