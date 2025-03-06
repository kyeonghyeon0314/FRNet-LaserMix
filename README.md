# 참고 모델
- [LaserMix](https://github.com/ldkong1205/LaserMix)
- [FRNet](https://github.com/Xiangxu-0103/FRNet)


# 추가해야할 사항들 
- 데이터셋에 맞는 data_root 조정
- mmdetection3d 설치
```
cd mmdetection3d
pip install -v -e .
```
- mmdetection의 mmdet3d를 지우고 lasermix의 mmdet3d로 대체(기존 mmdetection3d에는 model에 lasermix가 등록되어 있지 않음)

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

- FRNet의 최종 ```decode_head```로 ```FRhead```를 사용하는데 이는 ```['seg_logit']```으로 접근한다. 

결론적으로, ```lasermix.py```의 line112, 113의 ```'logits'```를 ```'seg_logit'```으로 변환하면 된다.


# MMdetection3D
[mean_teacher_hook](https://mmdetection.readthedocs.io/en/3.x/_modules/mmdet/engine/hooks/mean_teacher_hook.html)

현재 configs/lasermix_frnet/lasermix_frnet_smantickitti_seg.py 마지막 줄에 mmdet.MeanTeacherHook 을 부른 코드가 존재하는데 이는 EMA 업데이트를 위한 mmdetectio3d의 hook이다.
그러나 LaserMix의 mmdet 디렉토리에는 해당 코드가 존재하지 않고 mmdet/models/segmentors/lasermix.py 에서 mean_teacher 방식으로 EMA 형식으로 checkpoints 를 업데이트 하는것으로 확인된다. 아래 두 방법중 하나의 방법으로 설계할 예정
- 위 링크를 참조하여 ```mmdet/engine/hooks/mean_teacher_hook.py``` 추가 
- MMdetection3D 기준으로 teacher-student network 다시 설계
- lasermix.py를 따라 


# 학습및 시험
## train

19개의 학습 클래스를 Road, sidewalk, vehicle, unlabeled 총 4가지로 분류하여 재 학습한다.

```
python tools/create_data.py configs/lasermix_frnet/lasermix_frnet_semi_semantickitti_seg.py
```

## test

```
python test.py configs/lasermix_frnet/lasermix_frnet_semi_semantickitti_seg.py work_dirs/lasermix_frnet_semi_semantickitti_seg/best_miou_iter_18000.pth
```
