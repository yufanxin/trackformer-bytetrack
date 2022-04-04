"""
ByteTrack for TrackFormer

"""
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import clip_boxes_to_image, nms, box_iou

from ..util.box_ops import box_xyxy_to_cxcywh


class BYTETracker(object):
    """
    class responsible for tracking after detection
    """

    def __init__(self, obj_detector, obj_detector_post, tracker_cfg,
                 generate_attention_maps, logger=None) -> None:
        self._prev_blob = None
        self.obj_detector = obj_detector  # 目标检测 
        self.obj_detector_post = obj_detector_post  # 检测后处理, 主要把坐标转换成COCO格式
        # 一堆的阈值
        self.detection_obj_score_thresh = tracker_cfg['detection_obj_score_thresh']  # 检测高阈值
        self.detection_obj_score_low_thresh = tracker_cfg['detection_obj_score_low_thresh']  # 检测低阈值
        self.track_obj_score_thresh = tracker_cfg['track_obj_score_thresh']  # 跟踪阈值
        # NMS阈值
        self.detection_nms_thresh = tracker_cfg['detection_nms_thresh']
        self.track_nms_thresh = tracker_cfg['track_nms_thresh']

        self.public_detections = tracker_cfg['public_detections']  # False
        self.inactive_patience = float(tracker_cfg['inactive_patience'])  # 5
        self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']  # 0
        self.reid_sim_only = tracker_cfg['reid_sim_only']  # False
        self.generate_attention_maps = generate_attention_maps  # False
        self.reid_score_thresh = tracker_cfg['reid_score_thresh']  # Re-ID阈值
        self.reid_greedy_matching = tracker_cfg['reid_greedy_matching']  # False

        if self.generate_attention_maps:  # 暂时不用 不写
            pass

        self._logger = logger
        if self._logger is None:
            self._logger = lambda *log_strs: None

    @property
    # 返回object queries的数量
    def num_object_queries(self):
        return self.obj_detector.num_queries

    # 重置一切
    def reset(self, hard=True):
        self.tracks = []
        self.inactive_tracks = []
        self._prev_blob = None
        self._prev_frame = None

        if hard:
            self.track_num = 0
            self.results = {}
            self.frame_index = 0
            self.num_reids = 0

    @property
    def device(self):
        return next(self.obj_detector.parameters()).device

    def tracks_to_inactive(self, tracks):
        # 将tracks转换为inactive
        self.tracks = [t for t in self.tracks if t not in tracks]  # 不在tracks中的轨迹设置为inactive.换言之,只留下不在tracks中的轨迹

        for track in tracks:
            track.pos = track.last_pos[-1]  # 更新位置
        self.inactive_tracks += tracks  # 把tracks放入inactive(为什么?)

    def add_tracks(self, pos, scores, hs_embeds, masks=None, attention_maps=None):
        """Initializes new Track objects and saves them."""
        # 创建新轨迹并保存
        new_track_ids = []
        for i in range(len(pos)):  # 一共有len(pos)个新轨迹
            self.tracks.append(Track(  # 向self.tracks(list)中加入Track类 self.tracks中元素类别为class(Track)
                pos[i],  # 位置 应是[topleft_x,topleft_y,bottomright_x,bottomright_y]
                scores[i],  # 置信度
                self.track_num + i,  # 新的ID是已有轨迹数目加1
                hs_embeds[i],  # 可能是embedding?
                None if masks is None else masks[i],
                None if attention_maps is None else attention_maps[i],  # 给每个track也分配一个attention map
            ))
            new_track_ids.append(self.track_num + i)  # 存储新id的列表更新
        self.track_num += len(new_track_ids)  # 多了len(pos)个新轨迹

        if new_track_ids:
            self._logger(
                f'INIT TRACK IDS (detection_obj_score_thresh={self.detection_obj_score_thresh}): '
                f'{new_track_ids}')

        return new_track_ids

    def public_detections_mask(self, new_det_boxes, public_det_boxes):
        """Returns mask to filter current frame detections with provided set of
           public detections."""

        if not self.public_detections:  # public detections非空就返回检测框的size0大小的全是True的tensor(相当于没有mask)
            return torch.ones(new_det_boxes.size(0)).bool().to(self.device)

        else:
            raise NotImplementedError  # 暂时不考虑别的情况

    def reid(self, new_det_boxes, new_det_scores, new_det_hs_embeds,
             new_det_masks=None, new_det_attention_maps=None):
        # 根据当前提供的检测 从inactive的tracks中恢复

        self.inactive_tracks = [
            t for t in self.inactive_tracks
            if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]  # 如果t合法(面积大于0) 且t的生存周期还没到 更新self.inactive_tracks

        if not self.inactive_tracks or not len(new_det_boxes):  # 如果没有inactive tracks了
            return torch.ones(new_det_boxes.size(0)).bool().to(self.device)  # 返回一堆True

        dist_mat = []  # 距离矩阵

        if not self.reid_greedy_matching:
            # 以下应该是二部图匹配方式进行Re-ID
            for track in self.inactive_tracks:
                track_sim = track.hs_embed[-1]

                track_sim_dists = torch.cat([
                    F.pairwise_distance(track_sim, sim.unsqueeze(0))
                    for sim in new_det_hs_embeds])

                dist_mat.append(track_sim_dists)

            dist_mat = torch.stack(dist_mat)

            dist_mat = dist_mat.cpu().numpy()
            row_indices, col_indices = linear_sum_assignment(dist_mat)  # 获得匹配结果: 行->列
        else:
            raise NotImplementedError  # 贪心算法暂不考虑

        assigned_indices = []
        remove_inactive = []
        for row_ind, col_ind in zip(row_indices, col_indices):  # 遍历匹配结果
            if dist_mat[row_ind, col_ind] <= self.reid_sim_threshold:  # <=0???
                track = self.inactive_tracks[row_ind]

                self._logger(
                    f'REID: track.id={track.id} - '
                    f'count_inactive={track.count_inactive} - '
                    f'to_inactive_frame={self.frame_index - track.count_inactive}')

                track.count_inactive = 0
                track.pos = new_det_boxes[col_ind]
                track.score = new_det_scores[col_ind]
                track.hs_embed.append(new_det_hs_embeds[col_ind])
                track.reset_last_pos()

                if new_det_masks is not None:
                    track.mask = new_det_masks[col_ind]
                if new_det_attention_maps is not None:
                    track.attention_map = new_det_attention_maps[col_ind]

                assigned_indices.append(col_ind)
                remove_inactive.append(track)

                self.tracks.append(track)

                self.num_reids += 1

        for track in remove_inactive:
            self.inactive_tracks.remove(track)

        reid_mask = torch.ones(new_det_boxes.size(0)).bool().to(self.device)

        for ind in assigned_indices:
            reid_mask[ind] = False

        return reid_mask

    def step(self, blob):
        """
        采用ByteTrack的step, 无segm

        """
        # 重要 每一步都调用这个step函数 blob是含有图像信息的
        self._logger(f'FRAME: {self.frame_index + 1}')
        if self.inactive_tracks:
            self._logger(f'INACTIVE TRACK IDS: {[t.id for t in self.inactive_tracks]}')

        # add current position to last_pos list 将当前位置加入到最后位置中
        for track in self.tracks:
            track.last_pos.append(track.pos.clone())

        # 读取图像和原始尺寸
        img = blob['img'].to(self.device)
        orig_size = blob['orig_size'].to(self.device)

        target = None
        num_prev_track = len(self.tracks + self.inactive_tracks)  # num_prev_track是已有轨迹和inactive轨迹之和

        if num_prev_track:  # 创建track queries 的boxtensor target变量存着active和inactive的
            # boxes大小,id,embedding等信息
            track_query_boxes = torch.stack([
                t.pos for t in self.tracks + self.inactive_tracks], dim=0).cpu()  # 记录active和inactive的bbox

            track_query_boxes = box_xyxy_to_cxcywh(track_query_boxes)  # size表示的转换
            track_query_boxes = track_query_boxes / torch.tensor([
                orig_size[0, 1], orig_size[0, 0],
                orig_size[0, 1], orig_size[0, 0]], dtype=torch.float32)  # 除以原图高宽 作归一化

            target = {'track_query_boxes': track_query_boxes}

            target['image_id'] = torch.tensor([1]).to(self.device)  # image_id弄成1
            target['track_query_hs_embeds'] = torch.stack([
                t.hs_embed[-1] for t in self.tracks + self.inactive_tracks], dim=0)  # 记录head embeddings

            target = {k: v.to(self.device) for k, v in target.items()}  # 放入device
            target = [target]  # 将dict转换为list

        outputs, *_ = self.obj_detector(img, target)  # 将img和target放入detector 得到outputs

        hs_embeds = outputs['hs_embed'][0]

        det_results = self.obj_detector_post['bbox'](outputs, orig_size)  # 将输出转换为原始尺寸

        det_result = det_results[0]
        # result = {'scores': s, 'labels': l, 'boxes': b, 'scores_no_object': s_n_o}
        #     for s, l, b, s_n_o in zip(scores, labels, boxes, prob[..., -1])

        det_boxes = clip_boxes_to_image(det_result['boxes'], orig_size[0])
        # print(det_boxes.shape)

        '''
        上面做完的工作: 
        1.将过去所有的(active 和 inactive的)都用target表示. 在训练过程中, target是真值
        在推理过程中, target就是过去的所有轨迹. 可以当成ByteTrack里的Kalman结果.
        2.将当前图片放入检测模型输出outputs, 注意推理的时候Transformer的Tracking的forward不中用了, 实际上只剩下了检测.

        原来是先匹配老轨迹, 再匹配新检测. 现在可以在各自匹配过程中加入二次匹配, 或者干脆直接匹配.

        target就是T, result就是D!!
        '''

        # 第一步 首先筛选之前现有的轨迹 置信度低的先放弃 置信度高的匹配 
        # 置信度低的当成是新检测 直接Re-ID
        if num_prev_track:
            det_scores1 = det_result['scores'][:-self.num_object_queries]  # 取新检测中对应的老tracks
            det_boxes1 = det_boxes[:-self.num_object_queries]  # 和boxes
            hs_embeds1 = hs_embeds[:-self.num_object_queries]

            track_keep1_high = torch.logical_and(
                det_scores1 > self.track_obj_score_thresh,
                det_result['labels'][:-self.num_object_queries] == 0  # 目标类
            )  # 将det中老track的高阈值部分维持不动

            track_remain1 = []  # 没匹配的track放入track_remain1

            for i, track in enumerate(self.tracks):  # 高阈值第一次匹配
                if track_keep1_high[i]:  # 如果第i个老track是要保持的
                    # 更新参数
                    track.score = det_scores1[i]
                    track.hs_embed.append(hs_embeds[i])
                    track.pos = det_boxes1[i]

                else:  # 没跟det匹配的老track放入track_remain1
                    track_remain1.append(track)
            # print(f"****tracks and track remain 1 num{len(self.tracks)},,,,{len(track_remain1)}")
            self.tracks_to_inactive(track_remain1)  # 先放入inactive中 便于Re-ID
            # print(f"****inactive num{self.inactive_tracks}")

            track_keep1_low = torch.logical_and(
                det_scores1 > self.detection_obj_score_low_thresh,
                det_result['labels'][:-self.num_object_queries] == 0  # 目标类
            )  # 结果中阈值低的

            reid1_bbox, reid1_scores, reid1_hs_embeds = [], [], []

            track_remain2 = []
            for i, track in enumerate(self.tracks):
                if track_keep1_low[i] and not track_keep1_high[i]:

                    reid1_bbox.append(det_boxes1[i])
                    reid1_scores.append(det_scores1[i])
                    reid1_hs_embeds.append(hs_embeds1[i])

                elif not track_keep1_low[i]:
                    track_remain2.append(track)
            # print(f"****reid_low_thresh_len{len(reid1_bbox)}")
            # print(f"****track_remain{len(track_remain2)}")
            reid1_bbox = torch.tensor([item.cpu().detach().tolist() for item in reid1_bbox]).cuda()
            reid1_scores = torch.tensor([item.cpu().detach().tolist() for item in reid1_scores]).cuda()
            reid1_hs_embeds = torch.tensor([item.cpu().detach().tolist() for item in reid1_hs_embeds]).cuda()

            _ = self.reid(reid1_bbox, reid1_scores, reid1_hs_embeds, None, None)  # 在两阈值之间的Re-ID

            track_keep1_reid = torch.logical_and(  # 筛选恢复ID的tracks? 条件是大于Re-ID阈值 且 labels都是0
                det_scores1 > self.reid_score_thresh,
                det_result['labels'][:-self.num_object_queries] == 0)

            tracks_from_inactive = []

            # reid queries  在inactive里面找置信度大的
            for i, track in enumerate(self.inactive_tracks, start=len(self.tracks)):
                if track_keep1_reid[i]:  # 如果是要Re-ID的tracks 记录score embedding和pos
                    track.score = det_scores1[i]
                    track.hs_embed.append(hs_embeds[i])
                    track.pos = det_boxes1[i]

                    tracks_from_inactive.append(track)

            self.num_reids += len(tracks_from_inactive)
            # print(f"****tracks_from_inactive_len{len(tracks_from_inactive)}")
            for track in tracks_from_inactive:  # 从inactive中恢复track 就是从inactive中remove 再self.tracks中append回来
                self.inactive_tracks.remove(track)
                self.tracks.append(track)

            # self.tracks_to_inactive(track_remain2)  # 更新 tracks_to_inactive
            if self.track_nms_thresh and self.tracks:
                track_boxes = torch.stack([t.pos for t in self.tracks])
                track_scores = torch.stack([t.score for t in self.tracks])

                keep = nms(track_boxes, track_scores, self.track_nms_thresh)
                remove_tracks = [
                    track for i, track in enumerate(self.tracks)
                    if i not in keep]

                if remove_tracks:
                    self._logger(
                        f'REMOVE TRACK IDS (track_nms_thresh={self.track_nms_thresh}): '
                        f'{[track.id for track in remove_tracks]}')

                # self.tracks_to_inactive(remove_tracks)
                self.tracks = [
                    track for track in self.tracks
                    if track not in remove_tracks]
        # 接下来 处理新检测
        # 同理 先匹配高阈值
        det_scores2 = det_result['scores'][-self.num_object_queries:]
        det_boxes2 = det_boxes[-self.num_object_queries:]  # 和boxes
        hs_embeds2 = hs_embeds[-self.num_object_queries:]

        det_keep2_high = torch.logical_and(
            det_scores2 > self.detection_obj_score_thresh,
            det_result['labels'][-self.num_object_queries:] == 0  # 目标类
        )  # 高阈值对应索引

        # 筛选
        det_scores2_high = det_scores2[det_keep2_high]
        det_boxes2_high = det_boxes2[det_keep2_high]
        det_hs_embeds_high = hs_embeds2[det_keep2_high]

        # print(f"****new_det_high{det_boxes2_high.shape}")
        # Re-ID
        reid_mask2_high = self.reid(
            det_boxes2_high,
            det_scores2_high,
            det_hs_embeds_high,
            None, None)

        # 剩余的检测
        det_scores2_high_new = det_scores2_high[reid_mask2_high]
        det_boxes2_high_new = det_boxes2_high[reid_mask2_high]
        det_hs_embeds_high_new = det_hs_embeds_high[reid_mask2_high]

        # 筛选介于2阈值之间的
        det_keep2_low = torch.logical_and(
            det_scores2 > self.detection_obj_score_low_thresh,
            det_scores2 < self.detection_obj_score_thresh
        )

        det_keep2_low = torch.logical_and(
            det_keep2_low,
            det_result['labels'][-self.num_object_queries:] == 0
        )

        # 筛选
        det_scores2_low = det_scores2[det_keep2_low]
        det_boxes2_low = det_boxes2[det_keep2_low]
        det_hs_embeds_low = hs_embeds2[det_keep2_low]

        # 再Re-ID
        reid_mask2_low = self.reid(
            det_scores2_low,
            det_boxes2_low,
            det_hs_embeds_low,
            None, None
        )

        # 剩下的作为新轨迹
        det_scores_new = torch.cat((det_scores2_high_new, det_scores2_low[reid_mask2_low]), dim=0)
        det_boxes_new = torch.cat((det_boxes2_high_new, det_boxes2_low[reid_mask2_low]), dim=0)
        det_hs_embeds_new = torch.cat((det_hs_embeds_high_new, det_hs_embeds_low[reid_mask2_low]), dim=0)

        # print(f"****New_tracks_len{det_boxes_new.shape}")

        new_track_ids = self.add_tracks(
            det_boxes_new,
            det_scores_new,
            det_hs_embeds_new,
            None, None
        )
        # print(f"****new_track_ids{new_track_ids}")
        # NMS
        # 将新检测进行NMS筛选
        if self.detection_nms_thresh and self.tracks:
            track_boxes = torch.stack([t.pos for t in self.tracks])
            track_scores = torch.stack([t.score for t in self.tracks])

            new_track_mask = torch.tensor([
                True if t.id in new_track_ids
                else False
                for t in self.tracks])
            track_scores[~new_track_mask] = np.inf

            keep = nms(track_boxes, track_scores, self.detection_nms_thresh)
            remove_tracks = [track for i, track in enumerate(self.tracks) if i not in keep]

            if remove_tracks:
                self._logger(
                    f'REMOVE TRACK IDS (detection_nms_thresh={self.detection_nms_thresh}): '
                    f'{[track.id for track in remove_tracks]}')

            self.tracks = [track for track in self.tracks if track not in remove_tracks]  # 最终更新tracks

            # print(f"****remove_tracks_after_NMS{len(remove_tracks)}")
        ####################
        # Generate Results #
        ####################
        # 上面进行完轨迹处理(Re-ID,新检测等)后,下面要生成结果

        for track in self.tracks:  # 计算self.results 作为结果
            if track.id not in self.results:  # 如果是新id
                self.results[track.id] = {}  # 建立新id 初始化为空字典

            self.results[track.id][self.frame_index] = {}  # 初始化track.id的轨迹在第index帧为空字典
            self.results[track.id][self.frame_index]['bbox'] = track.pos.cpu().numpy()  # 加入bbox信息
            self.results[track.id][self.frame_index]['score'] = track.score.cpu().numpy()  # 加入置信度信息

        for t in self.inactive_tracks:
            t.count_inactive += 1  # 计算inactive的帧数

        self.frame_index += 1  # 帧索引加1 该计算下一帧了
        self._prev_blob = blob  # 将图像信息存入prev_blob

        if self.reid_sim_only:
            self.tracks_to_inactive(self.tracks)

    def get_results(self):  # 获得self.results
        """Return current tracking results."""
        return self.results


class Track(object):  # 每个轨迹的类 包含每个轨迹应有的信息
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, hs_embed,
                 mask=None, attention_map=None):
        self.id = track_id
        self.pos = pos
        self.last_pos = deque([pos.clone()])
        self.score = score
        self.ims = deque([])
        self.count_inactive = 0
        self.gt_id = None
        self.hs_embed = [hs_embed]
        self.mask = mask
        self.attention_map = attention_map

    def has_positive_area(self) -> bool:
        """Checks if the current position of the track has
           a valid, .i.e., positive area, bounding box."""
        return self.pos[2] > self.pos[0] and self.pos[3] > self.pos[1]

    def reset_last_pos(self) -> None:
        """Reset last_pos to the current position of the track."""
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())
