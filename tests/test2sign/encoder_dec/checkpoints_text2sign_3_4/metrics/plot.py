import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Load Data --------------------
data = pd.read_csv("metrics.csv")  # save your data as metrics.csv

# -------------------- Style --------------------
plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2
})

# -------------------- 1. Loss Plot --------------------
plt.figure()
plt.plot(data['epoch'], data['train_loss'], label='Train Loss')
plt.plot(data['epoch'], data['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_plot.png", dpi=300)
plt.show()

# -------------------- 2. Pose & Velocity Loss --------------------
plt.figure()
plt.plot(data['epoch'], data['val_pose_loss'], label='Pose Loss')
plt.plot(data['epoch'], data['val_velocity_loss'], label='Velocity Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Pose & Velocity Loss")
plt.legend()
plt.tight_layout()
plt.savefig("pose_velocity_loss.png", dpi=300)
plt.show()

# -------------------- 3. MPJPE & Velocity Error --------------------
plt.figure()
plt.plot(data['epoch'], data['val_mpjpe'], label='MPJPE')
plt.plot(data['epoch'], data['val_velocity_error'], label='Velocity Error')
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("MPJPE & Velocity Error")
plt.legend()
plt.tight_layout()
plt.savefig("error_metrics.png", dpi=300)
plt.show()

# -------------------- 4. Teacher Forcing & LR --------------------
fig, ax1 = plt.subplots()

ax1.plot(data['epoch'], data['teacher_forcing_ratio'], label='Teacher Forcing', linestyle='--')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Teacher Forcing Ratio")

ax2 = ax1.twinx()
ax2.plot(data['epoch'], data['lr'], label='Learning Rate', linestyle=':')
ax2.set_ylabel("Learning Rate")

fig.suptitle("Training Strategy Parameters")

fig.tight_layout()
plt.savefig("training_params.png", dpi=300)
plt.show()