# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp
from pathlib import Path

import mmengine

total_num = {  # semanticKITTI 데이터셋의 시퀀스 정보 : 스캔수
    0: 4541,
    1: 1101,
    2: 4661,
    3: 801,
    4: 271,
    5: 2761,
    6: 1101,
    7: 1101,
    8: 4071,
    9: 1591,
    10: 1201,
    11: 921,
    12: 1061,
    13: 3281,
    14: 631,
    15: 1901,
    16: 1731,
    17: 491,
    18: 1801,
    19: 4981,
    20: 831,
    21: 2721,
}
fold_split = {
    'train': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    'val': [8],
    'trainval': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'test': [0,1,2,3,4,5,6,7,8,9,10,11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}
split_list = ['train', 'valid', 'trainval', 'test']

def get_semantickitti_info(split):
    """Create info file in the form of
        data_infos={
            'metainfo': {'DATASET': 'SemanticKITTI'},
            'data_list': {
                00000: {
                    'lidar_points':{
                        'lidat_path':'sequences/00/velodyne/000000.bin'
                    },
                    'pts_semantic_mask_path':
                        'sequences/000/labels/000000.labbel',
                    'sample_id': '00'
                },
                ...
            }
        }
    """
    data_infos = dict()
    data_infos['metainfo'] = dict(dataset='SemanticKITTI')     # 데이터셋 이름 대문자로 적용중 
    data_list = []
    for i_folder in fold_split[split]:
        for j in range(total_num[i_folder]):
            data_list.append({
                'lidar_points': {
                    'lidar_path':
                    osp.join('sequences',
                             str(i_folder).zfill(2), 'velodyne',
                             str(j).zfill(6) + '.bin'),
                    'num_pts_feats':
                    4
                },
                'pts_semantic_mask_path':
                osp.join('sequences',
                         str(i_folder).zfill(2), 'labels',
                         str(j).zfill(6) + '.label'),
                'sample_id':
                #str(i_folder) + str(j)                     #lasermix의 기본 문자열 결합 방식 
                str(i_folder).zfill(2) + str(j).zfill(6)  #frnet의 zero padding 방식 mmdetection3D 프레임워크 방식 적용
            })
    data_infos.update(dict(data_list=data_list))
    return data_infos

# train_val 도 생성 mmdetection에서 사용권장
def create_semantickitti_info_file(pkl_prefix: str, save_path: str) -> None:
    print('Generate info.')
    save_path = Path(save_path)

    semantickitti_infos_train = get_semantickitti_info(split='train')
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'SemanticKITTI info train file is saved to {filename}')
    mmengine.dump(semantickitti_infos_train, filename)

    semantickitti_infos_val = get_semantickitti_info(split='val')
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'SemanticKITTI info val file is saved to {filename}')
    mmengine.dump(semantickitti_infos_val, filename)

    semantickitti_infos_trainval = get_semantickitti_info(split='trainval')
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'SemanticKITTI info trainval file is saved to {filename}')
    mmengine.dump(semantickitti_infos_trainval, filename)

    semantickitti_infos_test = get_semantickitti_info(split='test')
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'SemanticKITTI info test file is saved to {filename}')
    mmengine.dump(semantickitti_infos_test, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--root-path',
        type=str,
        default='./data/semantickitti',                         # default 경로 지정정
        help='specify the root path of dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/semantickitti',                          
        required=False,
        help='name of info pkl')
    parser.add_argument('--extra-tag', type=str, default='semantickitti')
    args = parser.parse_args()
    
    create_semantickitti_info_file(args.extra_tag, args.out_dir)
