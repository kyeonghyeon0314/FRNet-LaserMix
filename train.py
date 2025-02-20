# Copyright (c) OpenMMLab. All rights reserved.
# LaserMix 그대로 사용
import argparse  # 명령줄 인자 파싱
import logging   # 로깅 설정
import os        # 운영체제 인터페이스
import os.path as osp  # 경로 조작


from mmengine.config import Config, DictAction  # 구성 파일 관리
from mmengine.logging import print_log          # 로그 출력 유틸리티
from mmengine.registry import RUNNERS           # 러너 레지스트리
from mmengine.runner import Runner              # 주요 실행 클래스

from mmdet3d.utils import replace_ceph_backend  # Ceph 스토리지 지원

def parse_args():
    """명령줄 인자 파싱 함수"""
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    # 필수 인자: 구성 파일 경로
    parser.add_argument('config', help='train config file path')
    # 옵션 인자들
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(  # 자동 혼합 정밀도(AMP) 활성화
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(  # 학습률 자동 스케일링
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(  # 학습 재개 설정
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(  # Ceph 스토리지 사용
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(  # 런타임 구성 오버라이드
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(  # 분산 학습 환경 설정, GPU 하나만 사용시 필요없음
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # PyTorch >=2.0 호환성 처리
    # `torch.distributed.launch`가 `--local-rank` 대신 `--local_rank` 전달
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    
    args = parser.parse_args()
    # 분산 학습을 위한 환경 변수 설정
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()  # 명령줄 인자 파싱
    
    # 1. 구성 파일 로드
    cfg = Config.fromfile(args.config)
    
    # 2. Ceph 스토리지 백엔드 처리
    if args.ceph:
        cfg = replace_ceph_backend(cfg)  # Ceph 지원 활성화
    
    # 3. 기본 설정 적용
    cfg.launcher = args.launcher  # 실행 환경 설정
    
    # 4. 동적 구성 업데이트
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)  # CLI로 전달된 옵션 병합
    
    # 5. 작업 디렉토리 설정 우선순위:
    #    CLI > 구성 파일 > 기본값(구성 파일명 기반)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir  # CLI 값 우선
    elif cfg.get('work_dir', None) is None:
        # 구성 파일에 work_dir 없을 시 기본 경로 생성
        cfg.work_dir = osp.join('./work_dirs', 
                              osp.splitext(osp.basename(args.config))[0])
    
    # 6. AMP(자동 혼합 정밀도) 설정
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log('AMP가 이미 활성화됨', logger='current', level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                'AMP는 OptimWrapper에서만 지원됨')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'  # 래퍼 타입 변경
            cfg.optim_wrapper.loss_scale = 'dynamic'    # 동적 손실 스케일링
    
    # 7. 자동 학습률 조정
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
           'enable' in cfg.auto_scale_lr and \
           'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True  # 기능 활성화
        else:
            raise RuntimeError('자동 학습률 조정을 위한 설정 누락')
    
    # 8. 학습 재개 설정
    if args.resume == 'auto':
        cfg.resume = True    # 자동 재개 모드
        cfg.load_from = None # 최신 체크포인트 자동 탐색
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume  # 명시적 체크포인트 지정
    
    # 9. 러너 생성
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)  # 기본 러너 생성
    else:
        runner = RUNNERS.build(cfg)    # 커스텀 러너 생성
    
    # 10. 학습 시작
    runner.train()

if __name__ == '__main__':
    main()  # 프로그램 진입점
