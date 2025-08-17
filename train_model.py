# train_model_multioutput.py

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1) Load your cleaned season-level data
df = pd.read_csv("data/season_table.csv")

# 2) Helper to compute the “next” season string
def next_season(s):
    start, end = map(int, s.split('-'))
    return f"{start+1}-{end+1}"

# 3) Build training pairs from the past two seasons:
#    Features = [W, D, GF, GA] of season N
#    Targets  = same metrics of season N+1
train_seasons = ["2022-2023", "2023-2024"]
features      = ["W", "D", "GF", "GA"]

train_X, train_y = [], []
for season in train_seasons:
    ns = next_season(season)
    df_s  = df[df["Season"] == season]
    df_ns = df[df["Season"] == ns]
    for team in df_s["Team"]:
        if team in df_ns["Team"].values:
            x = df_s.loc[df_s["Team"] == team, features].values.flatten()
            y = df_ns.loc[df_ns["Team"] == team, features].values.flatten()
            train_X.append(x)
            train_y.append(y)

train_X = torch.tensor(train_X, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)

# 4) DataLoader
dataset = TensorDataset(train_X, train_y)
loader  = DataLoader(dataset, batch_size=16, shuffle=True)

# 5) Define a multi-output regression network
class MultiOutputNet(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_feats)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MultiOutputNet(len(features), len(features)).to(device)
opt    = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 6) Train
epochs = 200
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss  = loss_fn(preds, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{epochs} — Loss: {total_loss/len(dataset):.4f}")

# 7) Predict for 2024-2025 → 2025-2026
df_pred = df[df["Season"] == "2024-2025"].reset_index(drop=True)
Xp      = torch.tensor(df_pred[features].values, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    out = model(Xp).cpu().numpy().round().astype(int)

# 8) Build the full predicted table
pred = pd.DataFrame(out, columns=features)
pred["Team"] = df_pred["Team"]

# Derive L, GD, Pts from predictions
pred["L"]   = 38 - pred["W"] - pred["D"]
pred["GD"]  = pred["GF"] - pred["GA"]
pred["Pts"] = pred["W"] * 3 + pred["D"]

# Order and save
cols = ["Team", "W", "D", "L", "GF", "GA", "GD", "Pts"]
pred_table = pred[cols].sort_values("Pts", ascending=False).reset_index(drop=True)
pred_table.index = range(1, len(pred_table)+1)
pred_table.index.name = "Pos"

print("\n🔮 Predicted Premier League 2025/26 Full Standings:")
print(pred_table)
pred_table.to_csv("data/predicted_2025_26_full.csv")
print("\n✅ Saved to data/predicted_2025_26_full.csv")
