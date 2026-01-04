import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 填入推算出的四个数据：[[TN, FP], [FN, TP]]
cm = np.array([
    [25, 6],  # 第一行：真实标签为 HC (0) 的样本
    [0, 35]   # 第二行：真实标签为 PD (1) 的样本
])

# 2. 你的绘图代码
sns.set(style="white", font_scale=1.2)

plt.figure(figsize=(5.5, 5))
ax = sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    square=True,
    linewidths=1,
    linecolor="white",
    cbar=True,
    xticklabels=['HC', 'PD'], # 建议加上类别名称
    yticklabels=['HC', 'PD']
)

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("SC-EviMIL Confusion Matrix")

plt.tight_layout()

# 3. 保存并显示
plt.savefig("confusion_matrix_exp1.png", dpi=300)
print("图片已保存为 confusion_matrix_exp1.png")
plt.show()