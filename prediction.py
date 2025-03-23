import argparse
import os
import os.path as osp

from mmdet3d.utils import replace_ceph_backend
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

import os.path as osp
import tempfile
from typing import Dict, Optional, Sequence
import yaml
import mmcv
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
import mmengine
from mmdet3d.evaluation import seg_eval
from mmdet3d.registry import METRICS
import pdb
import torch

@METRICS.register_module()
class SemantickInferMertric(BaseMetric):
    
    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 result_path:str = None,
                 result_start_index:int = 0,
                 conf:str = None,
                 **kwargs):
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        self.result_path = result_path
        self.result_start_index = result_start_index
        self.current_start_index = self.result_start_index
        self.limit=[4541,1101,4661,801,271,2761,1101,1101,4071,1591,1201,921,1061,3281,631,1901,1731,491,1801,4981,831,2721]#the length of every test set seq
        self.limit_id = 0 #count 
        self.scene_id = 8 #seq id
        super(SemantickInferMertric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        self.conf = conf
        if self.conf :
            with open(self.conf) as f:
                self.conf = yaml.safe_load(f)
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        self.results.append((0, 0))

        # 현재 처리 중인 프레임 정보 출력
        print(f"\n====== 처리 중인 프레임 정보 ======")
        print(f"현재 시퀀스 ID: {self.scene_id}")
        print(f"현재 프레임 ID: {self.limit_id}")

        # 데이터 배치 구조 확인
        print(f"입력 데이터 키: {data_batch.keys()}")

        # 입력 데이터 정보 안전하게 출력 
        if 'inputs' in data_batch:
            if isinstance(data_batch['inputs'], dict):
                print(f"입력 데이터 타입: 딕셔너리")
                print(f"입력 데이터 키: {data_batch['inputs'].keys()}")
                print(f"입력 데이터 크기: {data_batch['inputs'].shape}")

        # label inv
        # pdb.set_trace()
        # pdb.set_trace()
        pred = data_samples[0]['pred_pts_seg']['pts_semantic_mask'] #labels
        print(f"원본 예측값: {pred.unique()}, 크기: {pred.shape}")
        map_inv = {
                0: 10,  # car
                1: 40,  # road
                2: 48,  # sidewalk
                3: 20,  # other-vehicle
                4: 0    # unlabeled
            }

        mapped_pred = torch.zeros_like(pred)

        # 각 클래스별로 매핑 적용
        for model_class, kitti_class in map_inv.items():
            mapped_pred[pred == model_class] = kitti_class
        
        print(f"매핑 후 예측값: {mapped_pred.unique()}, 크기: {mapped_pred.shape}")
        
        # 결과 저장
        work_dir = f'{self.result_path}/sequences/{self.scene_id}/predictions'
        os.makedirs(work_dir, exist_ok=True)
        
        mapped_pred.cpu().numpy().astype(np.int32).tofile(f'{work_dir}/{self.limit_id:06}.label')
        print(f'저장완료 : {self.scene_id}/predictions/{self.limit_id:06}.label')
        print(f"====== 프레임 처리 완료 ======\n")

        self.limit_id+=1

        if self.limit_id == self.limit[self.scene_id]: #next sequence
            print(f"시퀀스 {self.scene_id} 완료, 다음 시퀀스로 이동")
            self.scene_id += 1
            self.limit_id = 0
        # pdb.set_trace()

       

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        ret_dict = dict()

        return ret_dict

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D prediction script for 3D point cloud data')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument(
        '--work-dir', default='predictions/sequences',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--data-root', default='data/semantickitti',
        help='Path to the data root directory')
    parser.add_argument(
        '--sequences', nargs='+', default=None,
        help='Sequences to process (default: all sequences)')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--cuda', action='store_true', default=True,
        help='Use CUDA for inference')
    parser.add_argument(
        '--batch-size', type=int, default=1, 
        help='Batch size for inference (default: 1)')
    parser.add_argument(
        '--num-workers', type=int, default=1,
        help='Number of dataloader workers (default: 1)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    return args


def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    if args.ceph:
        cfg = replace_ceph_backend(cfg)
    
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.outpu_dir = osp.join('./work_dir',osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint
    
    # 데이터 루트 설정
    if args.data_root is not None:
        cfg.data_root = args.data_root

    #test dataloader 설정
    cfg.test_dataloader.dataset.data_root = cfg.data_root

    # 배치 크기 및 워커 수 설정
    cfg.test_dataloader.batch_size = args.batch_size
    cfg.test_dataloader.num_workers = args.num_workers

    # 테스트 평가기 설정
    cfg.test_evaluator = dict(
        type='SemantickInferMertric',
        result_path=cfg.work_dir,
    )

    # 런처 설정
    cfg.launcher = args.launcher

    # GPU 사용 설정
    if not args.cuda:
        cfg.device = 'cpu'

    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    
    runner.test()

    print(f'예측 완료! 결과가 {cfg.work_dir}에 저장되었습니다.')


if __name__ == '__main__':
    main()
