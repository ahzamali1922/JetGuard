import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
WINDOW_LENGTH = 50
RUL_CAP = 125.0


class TunedLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.recurrent = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.regression = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(sequence)
        return self.regression(self.dropout(outputs[:, -1])).squeeze(1)


class ModelArtifactError(RuntimeError):
    pass


class RULPredictor:
    def __init__(self) -> None:
        metadata_path = ARTIFACT_DIR / "lstm_tuning_metadata.json"
        scaler_path = ARTIFACT_DIR / "lstm_tuned_standard_scaler.joblib"
        checkpoint_path = ARTIFACT_DIR / "lstm_tuned_selected.pth"

        try:
            with metadata_path.open(encoding="utf-8") as metadata_file:
                self.metadata: dict[str, Any] = json.load(metadata_file)
            self.scaler = joblib.load(scaler_path)
            configuration = self.metadata["selected_configuration"]
            self.feature_names = list(self.scaler.feature_names_in_)
            if self.metadata["window_length"] != WINDOW_LENGTH:
                raise ValueError("The selected model must use a 50-cycle window.")
            if len(self.feature_names) != self.scaler.n_features_in_:
                raise ValueError("The saved scaler feature metadata is inconsistent.")

            self.model = TunedLSTM(
                input_size=len(self.feature_names),
                hidden_size=int(configuration["hidden_size"]),
                dropout=float(configuration["dropout"]),
            )
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ModelArtifactError(f"Unable to load the trained LSTM artifacts: {error}") from error

    def predict(self, observations: list[dict[str, Any]]) -> float:
        feature_rows = [
            [observation[name] for name in self.feature_names]
            for observation in observations
        ]
        values = pd.DataFrame(feature_rows, columns=self.feature_names, dtype=np.float64)
        values = self.scaler.transform(values)

        if len(values) >= WINDOW_LENGTH:
            values = values[-WINDOW_LENGTH:]
        else:
            padding = np.repeat(values[:1], WINDOW_LENGTH - len(values), axis=0)
            values = np.concatenate((padding, values), axis=0)

        sequence = torch.from_numpy(values.astype(np.float32, copy=False)).unsqueeze(0)
        with torch.inference_mode():
            prediction = float(self.model(sequence).item())
        return float(np.clip(prediction, 0.0, RUL_CAP))

    def public_info(self) -> dict[str, Any]:
        configuration = self.metadata["selected_configuration"]
        return {
            "model": "Tuned LSTM",
            "dataset": "NASA C-MAPSS FD001",
            "window_length": WINDOW_LENGTH,
            "target_cap_cycles": RUL_CAP,
            "input_features": self.feature_names,
            "selected_configuration": configuration,
            "official_test_metrics": self.metadata["official_test_metrics"],
        }


@lru_cache(maxsize=1)
def get_predictor() -> RULPredictor:
    return RULPredictor()
