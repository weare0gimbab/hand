import torch

# PyTorch 모델 로드
model = torch.load('runs\\detect\\train4\\weights\\best.pt')
model.eval()

# 더미 입력 생성 (모델 입력 크기에 맞게 조정)
dummy_input = torch.randn(1, 3, 224, 224)  # 예시: 1x3x224x224 크기

# ONNX로 내보내기
torch.onnx.export(model, dummy_input, 'model.onnx')
