tiny independent mobile sam

# prepare submodule
git submodule add https://gitee.com/yonggie/independent_mobile_sam.git ./mobile_sam

# use example
```py
import torch
import cv2
import numpy as np
from mobile_sam import SamPredictor, sam_model_registry

def load_mobilesam(checkpoint_path: str = "mobile_sam.pt", device="cuda"):
    model_type = "vit_t"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

# ===== 1. 创建一个 dummy 图像（例如：带一个白色方块的黑色背景）=====
height, width = 512, 512
dummy_image = np.zeros((height, width, 3), dtype=np.uint8)  # 黑色背景

# 在中间画一个白色方块（模拟一个物体）
cv2.rectangle(dummy_image, (200, 200), (300, 300), (255, 255, 255), -1)

# 转为 RGB（OpenCV 默认 BGR，但 MobileSAM 需要 RGB）
dummy_image_rgb = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)

# ===== 2. 加载模型 =====
predictor = load_mobilesam(
    "./mobile_sam.pt", 
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# ===== 3. 设置图像 =====
predictor.set_image(dummy_image_rgb)

# ===== 4. 定义点击点（点在白色方块中心附近）=====
input_point = np.array([[250, 250]])  # (x, y)
input_label = np.array([1])           # 1 = 前景点

# ===== 5. 推理 =====
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

# ===== 6. 保存掩码 =====
mask_binary = masks[0].astype(np.uint8) * 255  # 转为 0~255
cv2.imwrite("mask.png", mask_binary)

print("✅ Dummy 分割完成！掩码已保存为 mask.png")
print(f"   输入图像尺寸: {dummy_image_rgb.shape}")
print(f"   掩码形状: {masks[0].shape}")
```
