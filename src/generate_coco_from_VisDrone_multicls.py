'''
将Visdrone数据集转换为COCO格式 COCO格式详细说明:https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch

TrackFormer 训练指南 https://github.com/JackWoo0831/trackformer/blob/main/docs/TRAIN.md
'''

# import
import argparse
# import configparser
import csv
import json
import shutil

import os
import cv2

import skimage.io as io
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
import numpy as np

from matplotlib import pyplot as plt

# VisDrone路径
DATA_ROOT = '/data/wujiapeng/datasets/VisDrone2019/VisDrone2019/'
VIS_THRESHOLD = 0.0  # 认为是否忽略的阈值 默认为0

ignored_seqs = ['uav0000013_00000_v', 'uav0000013_01073_v', 'uav0000013_01392_v',
                'uav0000020_00406_v', 'uav0000071_03240_v', 'uav0000072_04488_v',
                'uav0000072_05448_v', 'uav0000072_06432_v', 'uav0000079_00480_v',
                'uav0000084_00000_v', 'uav0000099_02109_v', 'uav0000086_00000_v',
                'uav0000073_00600_v', 'uav0000073_04464_v', 'uav0000088_00290_v']


def generate_coco_from_visdrone(split_name='train', seqs_names=None,
                                root_split='VisDrone2019-MOT-train', frame_range=None, seq_range=None,
                                ignore_seqs=False):
    """

    """
    global DATA_ROOT

    if frame_range is None:  # 如果frame_range是空 就取全部帧 此处的数字是比例
        frame_range = {'start': 0.0, 'end': 1.0}

    root_split_path = os.path.join(DATA_ROOT, root_split, 'sequences')  # 数据读取的路径
    coco_dir = os.path.join(DATA_ROOT, split_name)

    # 将分割的data独立出来建立文件夹, 建立的是syslink(line )
    # if os.path.isdir(coco_dir): # 如果coco_dir存在就删除
    #    shutil.rmtree(coco_dir)

    os.mkdir(coco_dir)  # 再建立

    # annotations字典 最终将其写为json文件 包含数据集所有信息
    annotations = {}
    annotations['type'] = 'instances'
    annotations['images'] = []  # 图像集合
    # 类别 超类是vehicle 由于此处不作检测分类 故name不加区分
    annotations['categories'] = [{"supercategory": "vehicle", "name": "car", "id": 4},
                                 {"supercategory": "vehicle", "name": "van", "id": 5},
                                 {"supercategory": "vehicle", "name": "truck", "id": 6},
                                 {"supercategory": "vehicle", "name": "bus", "id": 9}]

    # 注意 原始Visdrone数据集标注的分类有12类
    '''annotations['categories'] = [
        {"supercategory": "ignored regions","name":"ignored regions","id":0},
        {"supercategory": "people","name":"pedestrain","id":1},
        {"supercategory": "people","name":"people","id":2},
        {"supercategory": "bicycle","name":"bicycle","id":3},
        {"supercategory": "vehicle","name":"car","id":4},
        {"supercategory": "vehicle","name":"van","id":5},
        {"supercategory": "vehicle","name":"truck","id":6},
        {"supercategory": "tricycle","name":"tricycle","id":7},
        {"supercategory": "tricycle","name":"awning-tricycle","id":8},
        {"supercategory": "vehicle","name":"bus","id":9},
        {"supercategory": "motor","name":"motor","id":10},
        {"supercategory": "others","name":"others","id":11}
    ]
    '''
    annotations['annotations'] = []  # annotations 最重要部分 包含每个图像里每个目标的信息

    annotations_dir = os.path.join(DATA_ROOT, 'annotations')  # 实际上是输出json的路径

    if not os.path.isdir(annotations_dir):  # 如不存在就创建
        os.mkdir(annotations_dir)

    annotation_file = os.path.join(annotations_dir, f'{split_name}.json')  # 对split_name的数据集转换成json文件的名称

    # 载入图片信息

    seqs = sorted(os.listdir(root_split_path))  # 序列的列表

    if seqs_names:  # 如果指定了转换的序列(一般用不到)
        seqs = [s for s in seqs if s in seqs_names]  # 筛选出指定的序列
    assert seqs != []

    if ignore_seqs:  # iditify the ignored seqs(usually contain the ignored object
        seqs = [s for s in seqs if s not in ignored_seqs]

    seq_range = float(seq_range)
    if seq_range < 1.0:  # if identify the range of seqs
        seqs = seqs[:int(seq_range * len(seqs))]

    # 增加信息 序列和帧范围
    annotations['sequences'] = seqs
    annotations['frame_range'] = frame_range

    img_id = 0  # 图像id 初始化为0

    print(f'Nums of seqs: {len(seqs)}')

    for seq in seqs:
        # 记录图片的信息 读取宽度 高度和序列长度
        # 每个sequence的图像长宽都应相同
        images_in_seq = os.listdir(os.path.join(root_split_path, seq))
        img_eg = cv2.imread(os.path.join(root_split_path, seq, images_in_seq[0]))

        img_width, img_height = img_eg.shape[1], img_eg.shape[0]

        seq_length = len(images_in_seq)

        seg_list_dir = os.listdir(os.path.join(root_split_path, seq))  # 读取该seq下的图片
        start_frame = int(frame_range['start'] * seq_length)  # 计算范围的开始帧
        end_frame = int(frame_range['end'] * seq_length)  # 计算范围的结束帧
        seg_list_dir = seg_list_dir[start_frame: end_frame]  # 提取

        for i, img in enumerate(sorted(seg_list_dir)):

            if i == 0:
                first_frame_image_id = img_id  # 本序列中第一帧的图像id起始

            # 加入images信息
            annotations['images'].append({"file_name": f"{seq}_{img}",
                                          "height": img_height,
                                          "width": img_width,
                                          "id": img_id,
                                          "frame_id": i,
                                          "seq_length": seq_length,
                                          "first_frame_image_id": first_frame_image_id})

            img_id += 1  # 下一张图片id加1

            os.symlink(os.path.join(os.getcwd(), root_split_path, seq, img),
                       os.path.join(coco_dir, f"{seq}_{img}"))

    # 读取GT 完成annotations['annotations']部分

    # 为每个image都创建annotation 因此建立image_name:image_id字典
    img_file_name_to_id = {
        img_dict['file_name']: img_dict['id']
        for img_dict in annotations['images']}

    annotation_id = 0
    for seq in seqs:

        gt_file_path = os.path.join(DATA_ROOT, root_split, 'annotations', seq + '.txt')  # GT文件路径
        if not os.path.isfile(gt_file_path):  # 以防万一没有
            raise FileNotFoundError
            # continue

        seq_annotations = []  # 本seq中记录images的annotation

        with open(gt_file_path, "r") as gt_file:
            reader = csv.reader(gt_file, delimiter=',')  # 读取GT文件

            for row in reader:
                if row[6] == '1' and row[7] in ['4', '5', '6', '9']:  # score部分为1,评估时考虑边界框 并且目标的类别为车辆

                    bbox = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]  # 读取边界框
                    bbox = [int(c) for c in bbox]  # 转换为int
                    area = bbox[2] * bbox[3]  # 面积

                    visibility = 1 - float(row[9]) / 3  # 遮挡程度
                    # visibility = 1

                    frame_id = int(row[0])  # 帧id 也对应image id
                    # 从帧id中获取image id
                    image_id = img_file_name_to_id.get(f"{seq}_{frame_id:07d}.jpg", None)
                    if image_id is None:
                        continue

                    track_id = int(row[1])  # 目标ID值

                    # choose class
                    if row[7] == '4':
                        cat_id = annotations['categories'][0]['id']
                    elif row[7] == '5':
                        cat_id = annotations['categories'][1]['id']
                    elif row[7] == '6':
                        cat_id = annotations['categories'][2]['id']
                    else:
                        cat_id = annotations['categories'][3]['id']


                    annotation = {
                        "id": annotation_id,
                        "bbox": bbox,
                        "image_id": image_id,
                        "segmentation": [],
                        "ignore": 0 if visibility > VIS_THRESHOLD else 1,
                        "visibility": visibility,
                        "area": area,
                        "iscrowd": 0,
                        "seq": seq,
                        "category_id": cat_id,
                        "track_id": track_id}

                    seq_annotations.append(annotation)

                    annotation_id += 1

        annotations['annotations'].extend(seq_annotations)

    # 写json文件
    with open(annotation_file, 'w') as anno_file:
        json.dump(annotations, anno_file, indent=4)  # indent:缩进


def check(split_name='train'):
    """
    Visualize generated COCO data. Only used for debugging.
    """
    coco_dir = os.path.join(DATA_ROOT, split_name)
    annotation_file = os.path.join(DATA_ROOT, 'annotations', f'{split_name}.json')

    coco = COCO(annotation_file)
    cat_ids = coco.getCatIds(catNms=['vehicle'])
    img_ids = coco.getImgIds(catIds=cat_ids)

    index = np.random.randint(0, len(img_ids))
    img = coco.loadImgs(img_ids[index])[0]

    i = io.imread(os.path.join(coco_dir, img['file_name']))

    plt.imshow(i)
    plt.axis('off')
    ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    coco.showAnns(anns, draw_bbox=True)
    plt.savefig('annotations.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate COCO from VisDrone.')
    parser.add_argument('--split_name', default='train_coco')
    parser.add_argument('--root_split', default='VisDrone2019-MOT-train')
    parser.add_argument('--seq_names', default=None)
    parser.add_argument('--frame_range', default=None)
    parser.add_argument('--seq_range', default=1.0)
    parser.add_argument('--cross_val', default=False)
    parser.add_argument('--ignore_seqs', default=True)
    parser.add_argument('--check', default=True)

    args = parser.parse_args()

    generate_coco_from_visdrone(split_name=args.split_name, seqs_names=args.seq_names
                                , root_split=args.root_split, frame_range=args.frame_range, seq_range=args.seq_range,
                                ignore_seqs=args.ignore_seqs)
    if args.check:
        check(args.split_name)
    # TODO
    if args.cross_val:
        raise NotImplementedError

    print('Done!')

    '''
    以后可能会加入交叉验证 想办法实现随机抽取序列作为训练和验证

    '''