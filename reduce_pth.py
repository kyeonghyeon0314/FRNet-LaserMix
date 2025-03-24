import torch

# 전체 체크포인트 로드
checkpoint = torch.load('work_dirs/best_miou_iter_148500.pth')

# 모델 가중치만 추출
weights_only = checkpoint['state_dict']

# 새 파일로 저장
torch.save(weights_only, 'weights_only_148500.pth')
