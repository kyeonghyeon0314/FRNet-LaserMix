# 참고 모델
- [LaserMix](https://github.com/ldkong1205/LaserMix)
- [FRNet](https://github.com/Xiangxu-0103/FRNet)


# 추가해야할 사항들 
- 데이터셋에 맞는 data_root 조정
- mmdetection3d 설치
- mmdetection의 mmdet3d를 지우고 lasermix의 mmdet3d로 대체(기존 mmdetection3d에는 model에 lasermix가 등록되어 있지 않음)
- [LaserMix/DATA_PREPARE.MD](https://github.com/ldkong1205/LaserMix/blob/main/docs/DATA_PREPARE.md)에 나오있는 [링크](https://drive.google.com/drive/folders/1PInw2Wvt-vgNzOxlSd2EiDANrTsWV7w1)에서 ```.pkl```을 다운 받고 아래 디렉토리 경로로 이동

```
└── semantickitti
    ├── sequences
    ├── semantickitti_infos_train.pkl
    ├── semantickitti_infos_val.pkl
    ├── ...
    ├── semantickitti_infos_train.10.pkl
    ├── semantickitti_infos_train.10-unlabeled.pkl
    └── ...
```
추가 내용, LaserMix README 파일을 보면 mmdetection3d에 등록되었다고하는데 이는 데이터 증강 알고리즘의 LaserMix로 mmdetection3d/mmdet3d/datasets/transforms/transforms_3d.py 클래스로 존재

# 모델 변환으로 인한 수정사항
## LaserMix의 loss 함수 분기 처리
```mmdet3d/models/segmentor/lasermix.py```의 loss 함수는 입력 형식에 따라 두 분기로 나누어진다.

- ```multi_batch_inputs['sup']```안에 'imgs'키가 존재한다면
 -> ```"range view"``` 분기가 실행되며, teacher/student 출력은 tensor로 반환되어 바로 torch.cat()을 사용한다.

- ```multi_batch_inputs['sup']```안에 ```'imgs'```키가 존재하지 않다면
 -> ```"else"``` 분기가 실행되고, 여기서는 teacher/student 출력이 dict 형태로 반환되어 ```['logits']```로 접근하도록 작성되어 있다.

## FRNet 출력 형식 확인
- 기존 LaserMix 백본 Cylinder3D는 dictionary 형태로 반환되어 ```['logits']```으로 접근했지만,

- FRNet의 ```decode_head```로 ```FRhead```를 사용하는데 이는 ```['seg_logit']```으로 접근한다. 

결론적으로, ```lasermix.py```의 line112, 113의 logits를 seg_logit으로 변환하면 된다.

# 실행
```
python tools/create_data.py configs/lasermix_frnet/lasermix_frnet_semi_semantickitti_seg.py
```
우선 서버에 학습을 실행중이다. 성능 검증은 추후 학습이 완료되면 해볼 예정

