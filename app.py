import streamlit as st
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import timm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import gdown

# --- CẤU HÌNH MÔ HÌNH (BẮT BUỘC GIỐNG KHI TRAIN) ---
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class InvariantModel(nn.Module):
    def __init__(self, num_disease_classes=2, num_domains=3):
        super(InvariantModel, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0, global_pool='avg')
        num_features = self.backbone.num_features
        self.disease_head = nn.Sequential(
            nn.Linear(num_features, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_disease_classes), nn.Softplus()
        )
        self.domain_head = nn.Sequential(
            nn.Linear(num_features, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_domains)
        )
    def forward(self, x, alpha=1.0):
        features = self.backbone(x)
        evidence = self.disease_head(features)
        return evidence

# --- HÀM TẢI MÔ HÌNH ---
@st.cache_resource
def load_model():
    model_path = "invariant_melanoma_model.pth"
    if not os.path.exists(model_path):
        url = f'https://drive.google.com/uc?id=11kKrUoQwChLgUjNAZud41Vdcv2GuC1ou'
        gdown.download(url, model_path, quiet=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InvariantModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="IRM-CRC Skin Cancer Diagnosis", layout="centered")
st.title("🔬 Hệ thống Chẩn đoán Ung thư Da Kháng Ảo giác")
st.markdown("""
Ứng dụng sử dụng kiến trúc **Domain-Invariant (IRM)** để triệt tiêu sai lệch thiết bị 
và **Conformal Risk Control (CRC)** để đảm bảo an toàn lâm sàng.
""")

# Thiết lập tham số cố định từ quá trình Hiệu chuẩn (Calibration)
Q_HAT = 0.4368 

model, device = load_model()

uploaded_file = st.file_uploader("Tải lên ảnh chụp nội soi da (Dermoscopy)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
    
    # Tiền xử lý
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    img_array = np.array(image)
    augmented = transform(image=img_array)
    img_tensor = augmented['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        evidence = model(img_tensor)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = (alpha / S).cpu().numpy()[0]
        uncertainty = (2 / S).cpu().numpy()[0][0]

    # Kiểm soát rủi ro (CRC Logic)
    prediction_set = []
    if probs[0] >= Q_HAT: prediction_set.append("Lành tính (Benign)")
    if probs[1] >= Q_HAT: prediction_set.append("Ác tính (Melanoma)")

    # Hiển thị kết quả
    st.subheader("Kết quả phân tích:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Xác suất Lành tính", f"{probs[0]*100:.2f}%")
        st.metric("Xác suất Ác tính", f"{probs[1]*100:.2f}%")
    
    with col2:
        st.metric("Độ bất định (Uncertainty)", f"{uncertainty:.4f}")
        if len(prediction_set) == 1:
            st.success(f"Chẩn đoán: {prediction_set[0]}")
        else:
            st.warning("⚠️ Cảnh báo: Hệ thống không đủ bằng chứng để phân loại đơn lẻ. Cần kiểm tra lâm sàng thêm.")

    st.info(f"Ngưỡng an toàn toán học (q_hat) đang áp dụng: {Q_HAT}")
