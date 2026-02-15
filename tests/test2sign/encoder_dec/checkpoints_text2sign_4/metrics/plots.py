import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("epoch_metrics.csv")  
os.makedirs("plots", exist_ok=True)

# ---------- Loss curves ----------
plt.figure()
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/loss_curve.png", dpi=300)
plt.close()

# ---------- MPJPE ----------
plt.figure()
plt.plot(df["epoch"], df["val_mpjpe"])
plt.xlabel("Epoch")
plt.ylabel("MPJPE")
plt.title("Validation MPJPE")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/val_mpjpe.png", dpi=300)
plt.close()

# ---------- Velocity Error ----------
plt.figure()
plt.plot(df["epoch"], df["val_velocity_error"])
plt.xlabel("Epoch")
plt.ylabel("Velocity Error")
plt.title("Validation Velocity Error")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/val_velocity_error.png", dpi=300)
plt.close()

# ---------- Teacher Forcing Ratio ----------
plt.figure()
plt.plot(df["epoch"], df["teacher_forcing_ratio"])
plt.xlabel("Epoch")
plt.ylabel("Teacher Forcing Ratio")
plt.title("Teacher Forcing Schedule")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/teacher_forcing_ratio.png", dpi=300)
plt.close()

# ---------- Learning Rate ----------
plt.figure()
plt.plot(df["epoch"], df["lr"])
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/learning_rate.png", dpi=300)
plt.close()

print("✅ All plots saved in ./plots/")
