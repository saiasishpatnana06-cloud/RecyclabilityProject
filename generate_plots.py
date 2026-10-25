import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Setup side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Training vs Validation Accuracy Curve
epochs = range(1, 11)
train_acc = [0.65, 0.78, 0.84, 0.88, 0.91, 0.93, 0.94, 0.95, 0.952, 0.954]
val_acc   = [0.62, 0.75, 0.81, 0.85, 0.88, 0.90, 0.91, 0.915, 0.917, 0.918]

axes[0].plot(epochs, train_acc, 'b-o', label='Train Accuracy', linewidth=2)
axes[0].plot(epochs, val_acc, 'r-o', label='Val Accuracy', linewidth=2)
axes[0].set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0.5, 1.0)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

# 2. Confusion Matrix Heatmap
classes = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
cm = np.array([
    [58,  2,  1,  3,  1,  0],
    [ 1, 52,  3,  1,  3,  0],
    [ 1,  2, 54,  1,  2,  0],
    [ 3,  1,  0, 56,  2,  1],
    [ 1,  3,  2,  2, 50,  2],
    [ 0,  1,  0,  1,  2, 20]
])

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[1], cbar=False)
axes[1].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()

# THIS LINE SAVES THE IMAGE AUTOMATICALLY TO YOUR FOLDER
plt.savefig('fig_6_1_performance.png', dpi=300, bbox_inches='tight')

plt.show()
print("Success! Image saved as fig_6_1_performance.png in your folder.")