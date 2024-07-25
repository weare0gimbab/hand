import torch
import torch.nn as nn

# 모델 클래스 정의 (예시)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 2)  # 예시 레이어

    def forward(self, x):
        return self.fc(x)

# 모델 인스턴스 생성
model = MyModel()

# 저장된 checkpoint 로드
checkpoint = torch.load('C:\\Users\\joeup\\Desktop\\weare0\\hand\\runs\\detect\\train4\\weights\\best.pt')

# 모델의 state_dict 업데이트
model.load_state_dict(checkpoint['model'])  # 'model' 키로 접근

# 평가 모드로 전환
model.eval()

print("모델이 성공적으로 로드되었습니다.")
