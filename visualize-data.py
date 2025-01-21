from torch.utils.data import DataLoader
from RGBathy import RGBathy
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np


def load_config(path_to_config):
    """
    Funktion zum Einlesen einer YAML-Konfigurationsdatei.
    """
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)
    return config



# 1) Config laden
config = load_config("configs/config-example.yaml")

# 2) Dataset-Objekte für train, val und test erstellen
train_dataset = RGBathy(config, split_mode="train")

train_loader = DataLoader(train_dataset,
                            batch_size=1, # config["batch_size"],
                            shuffle=True,  # Shuffle nur bei Training
                            num_workers=0,  # [Optimierungspunkt] Anzahl der Worker erhöhen
                            pin_memory=True) # [Optimierungspunkt] Nutzen bei GPU

# 3) Visualisierung der Daten
fig, axs = plt.subplots(4, 2, figsize=(10, 20))

axs[0, 0].set_title("Image")
axs[0, 1].set_title("Label")


for batch_idx, (inputs, labels) in enumerate(train_loader):

    inputImg = inputs[0].permute(1,2,0).numpy()
    axs[batch_idx, 0].imshow(inputImg, aspect="auto")
    axs[batch_idx, 0].axis("off")

    label = labels[0].permute(1,2,0).numpy()
    axs[batch_idx, 1].imshow(label, aspect="auto")
    axs[batch_idx, 1].axis("off")

    if batch_idx == 3:
        break

plt.tight_layout()
plt.savefig(f"./visualization.png", dpi=300)