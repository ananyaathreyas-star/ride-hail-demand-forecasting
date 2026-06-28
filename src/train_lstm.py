import argparse

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import MODELS_DIR
from src.data_pipeline import load_hourly_demand


class DemandLSTM(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def make_sequences(values: np.ndarray, sequence_length: int):
    X, y = [], []
    for i in range(sequence_length, len(values)):
        X.append(values[i - sequence_length : i])
        y.append(values[i])
    return np.array(X), np.array(y)


def train_lstm(zone_id: int = 8, sequence_length: int = 24, epochs: int = 10, batch_size: int = 32):
    df = load_hourly_demand()
    zone = df[df["zone_id"] == zone_id].sort_values("datetime_hour")
    if len(zone) < sequence_length + 50:
        raise ValueError(f"Not enough rows for zone {zone_id}. Try a larger download or another zone.")

    values = zone["trip_count"].astype(float).values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values).flatten()
    X, y = make_sequences(scaled, sequence_length)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = DemandLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"epoch={epoch + 1} loss={np.mean(losses):.4f}")

    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)).numpy()

    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    metrics = {
        "zone_id": zone_id,
        "mae": float(mean_absolute_error(actual, preds)),
        "rmse": float(np.sqrt(mean_squared_error(actual, preds))),
        "test_rows": int(len(actual)),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sequence_length": sequence_length,
            "zone_id": zone_id,
            "metrics": metrics,
        },
        MODELS_DIR / f"lstm_zone_{zone_id}.pt",
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PyTorch LSTM for one pickup zone.")
    parser.add_argument("--zone-id", type=int, default=8, help="Chicago community area ID. 8 is Near North Side.")
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    print(train_lstm(args.zone_id, args.sequence_length, args.epochs))
