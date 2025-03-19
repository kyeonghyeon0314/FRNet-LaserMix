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
## mmdetection/mmdet/datasets/semantickitti-dataset.py METAINFO 수정
labels_map에 보면 bus는 other-vehicle로 맵핑 되어있지만 METAINFO에는 bus가 존재 [semantickitti-api의 yaml](https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml) 참고하여 번호 순서에 맞게 MEATAINFO 수정



# 학습및 시험
## train

- 기존 MMdetection의 19개의 학습 클래스를 Road, sidewalk, car, other-vehicle, unlabeled 총 5가지로 분류
- teacher-network에 사전 학습된 checkpoints 적용 

```
python train.py configs/lasermix_frnet/lasermix_frnet_semantickitti_seg.py
```

## test

```
python test.py configs/lasermix_frnet/lasermix_frnet_semi_semantickitti_seg.py work_dirs/lasermix_frnet_semi_semantickitti_seg/best_miou_iter_18000.pth
```

## 최종 결과
- car를 제외한 나머지 vehicle을 other-vehicle로 묶고 향상된 학습법으로 학습했을시, **IoU**(67.45, 61.93) 점수가 **1.87%, 7.39%** 향상한것을 보였습니다.
- 최종적으로 논문의 ***mIoU*** 점수(74.69, 84.75) 보다 약 **13%, 2.92%** 향상한 **87.67%** 의 결과를 보였습니다.
<div align="center">
  <div style="margin-bottom: 10px;">
    <img src="/imgs/origin.png" width="90%">
    <p style="text-align: center;">논문 결과</p>
  </div>
</div>

- 제공된 Check points의 test 결과

<div align="center">
  <div style="margin-bottom: 10px;">
    <img src="/imgs/checkpoints.png" width="90%">
    <p style="text-align: center;">제공된 check points 결과</p>
  </div>
</div>

- 학습완료한 Check points의 test 결과

<div align="center">
  <div style="margin-bottom: 10px;">
    <img src="/imgs/test.png" width="80%">
    <p style="text-align: center;">최종 학습된 check points 결과</p>
  </div>
</div>
