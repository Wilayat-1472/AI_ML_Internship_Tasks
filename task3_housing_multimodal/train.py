"""
Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular Data
Predict housing prices using both structured data and house images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = "./housing_data"
MODEL_DIR = "./multimodal_model"
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Task 3: Multimodal Housing Price Prediction")
print("=" * 60)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("\n[1/6] Generating synthetic housing dataset...")
n_samples = 2000

areas = np.random.uniform(500, 5000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.uniform(0, 50, n_samples)
garage = np.random.randint(0, 3, n_samples)
location_score = np.random.uniform(1, 10, n_samples)

prices = (
    areas * 150
    + bedrooms * 25000
    + bathrooms * 15000
    - age * 1000
    + garage * 20000
    + location_score * 30000
    + np.random.normal(0, 20000, n_samples)
)

df = pd.DataFrame({
    "area": areas, "bedrooms": bedrooms, "bathrooms": bathrooms,
    "age": age, "garage": garage, "location_score": location_score,
    "price": prices,
})

print(f"  Generated {n_samples} samples")
print(f"  Price range: ${prices.min():,.0f} - ${prices.max():,.0f}")
print(f"  Mean price: ${prices.mean():,.0f}")

print("\n[2/6] Generating synthetic house images...")
img_dir = os.path.join(DATA_DIR, "images")
os.makedirs(img_dir, exist_ok=True)

def generate_house_image(idx, area, bedrooms, price):
    img = np.random.randint(180, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    n_stories = 1 if float(area) < 2000 else 2
    wall_color = np.array([np.random.randint(150, 230), np.random.randint(120, 200), np.random.randint(80, 160)])
    for y in range(40, 90):
        for x in range(20, 110):
            img[y, x] = wall_color
    roof_color = np.array([np.random.randint(80, 140), np.random.randint(40, 80), np.random.randint(30, 60)])
    for y in range(20, 45):
        for x in range(15, 115):
            if abs(y - 42) > abs(x - 65) * 0.3:
                img[y, x] = roof_color
    door_color = np.array([101, 67, 33])
    for y in range(60, 88):
        for x in range(55, 70):
            img[y, x] = door_color
    for i in range(min(int(bedrooms), 3)):
        wx = 25 + i * 30
        for y in range(50, 65):
            for x in range(wx, wx + 15):
                img[y, x] = np.array([135, 206, 235])
    img_pil = Image.fromarray(img)
    img_pil.save(os.path.join(img_dir, f"house_{idx}.png"))

for i in range(n_samples):
    generate_house_image(i, df.iloc[i]["area"], df.iloc[i]["bedrooms"], df.iloc[i]["price"])

print(f"  Generated {n_samples} house images ({IMG_SIZE}x{IMG_SIZE})")

print("\n[3/6] Preparing dataset and dataloaders...")
feature_cols = ["area", "bedrooms", "bathrooms", "age", "garage", "location_score"]
X_tab = df[feature_cols].values
y = df["price"].values

scaler = StandardScaler()
X_tab = scaler.fit_transform(X_tab)
y_log = np.log1p(y)

X_tab_train, X_tab_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_tab, y_log, np.arange(n_samples), test_size=0.2, random_state=42
)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class HousingDataset(Dataset):
    def __init__(self, tabular, targets, indices, img_dir, transform=None):
        self.tabular = tabular
        self.targets = targets
        self.indices = indices
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = Image.open(os.path.join(self.img_dir, f"house_{real_idx}.png")).convert("RGB")
        if self.transform:
            img = self.transform(img)
        tab = torch.tensor(self.tabular[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return img, tab, target


train_ds = HousingDataset(X_tab_train, y_train, idx_train, img_dir, train_transform)
test_ds = HousingDataset(X_tab_test, y_test, idx_test, img_dir, test_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

print("\n[4/6] Building multimodal model (CNN + Tabular)...")


class MultimodalHousingModel(nn.Module):
    def __init__(self, tabular_dim=6, hidden_dim=128):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        for param in list(self.cnn.parameters())[:-20]:
            param.requires_grad = False
        self.cnn.fc = nn.Identity()
        cnn_out = 512

        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + 32, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, img, tab):
        img_feat = self.cnn(img)
        tab_feat = self.tabular_net(tab)
        combined = torch.cat([img_feat, tab_feat], dim=1)
        return self.fusion(combined).squeeze(-1)


model = MultimodalHousingModel().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params: {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")
print(f"  Device: {DEVICE}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

print("\n[5/6] Training...")
train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for img, tab, target in train_loader:
        img, tab, target = img.to(DEVICE), tab.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        pred = model(img, tab)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * img.size(0)
    train_loss = running_loss / len(train_ds)
    train_losses.append(train_loss)

    model.eval()
    test_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for img, tab, target in test_loader:
            img, tab, target = img.to(DEVICE), tab.to(DEVICE), target.to(DEVICE)
            pred = model(img, tab)
            test_loss += criterion(pred, target).item() * img.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    test_loss /= len(test_ds)
    test_losses.append(test_loss)
    scheduler.step(test_loss)

    preds_original = np.expm1(all_preds)
    targets_original = np.expm1(all_targets)
    mae = mean_absolute_error(targets_original, preds_original)
    rmse = np.sqrt(mean_squared_error(targets_original, preds_original))

    print(f"  Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | MAE: ${mae:,.0f} | RMSE: ${rmse:,.0f}")

print("\n[6/6] Final evaluation...")
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for img, tab, target in test_loader:
        img, tab, target = img.to(DEVICE), tab.to(DEVICE), target.to(DEVICE)
        pred = model(img, tab)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

preds_original = np.expm1(all_preds)
targets_original = np.expm1(all_targets)
final_mae = mean_absolute_error(targets_original, preds_original)
final_rmse = np.sqrt(mean_squared_error(targets_original, preds_original))
ss_res = np.sum((targets_original - preds_original) ** 2)
ss_tot = np.sum((targets_original - np.mean(targets_original)) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

print(f"\n  Final MAE:  ${final_mae:,.0f}")
print(f"  Final RMSE: ${final_rmse:,.0f}")
print(f"  R² Score:   {r2:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Curves")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(targets_original, preds_original, alpha=0.3, s=10)
min_val = min(targets_original.min(), preds_original.min())
max_val = max(targets_original.max(), preds_original.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title(f"Actual vs Predicted (R²={r2:.4f})")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "results.png"), dpi=150)

torch.save({
    "model_state_dict": model.state_dict(),
    "scaler": scaler,
    "metrics": {"mae": float(final_mae), "rmse": float(final_rmse), "r2": float(r2)},
}, os.path.join(MODEL_DIR, "multimodal_model.pt"))

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump({"mae": float(final_mae), "rmse": float(final_rmse), "r2": float(r2)}, f, indent=2)

print(f"\n  Model saved to {MODEL_DIR}")
print("=" * 60)
print("Task 3 COMPLETE")
print("=" * 60)
