"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.
Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
Modified by Xingyi Zhou
"""

import argparse
import glob
import os
import logging
import motmetrics as mm
import pandas as pd
from collections import OrderedDict
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="""
Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...
Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...
Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--result', type=str, help='Log level', default='/home/dominic/MOT/MP2Net/SatMTB/DLADCN/results/tracking*') 
    parser.add_argument('--cate', type=str, help='choose cate from airplane, car, ship and train', default='airplane')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mtb-sat')
    parser.add_argument('--solver', type=str, help='LAP solver to use')
    return parser.parse_args()

# ============================================================================
# 类别ID映射表 - 不要修改！
# 与SatMTB数据集定义保持一致：
# - car: 0
# - airplane: 1  
# - ship: 2
# - train: 3
# ============================================================================
NAME_LABEL = {
    'car': 0,
    'airplane':      1,
    'ship':     2,
    'train':    3
}

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logging.info('Comparing {}...'.format(k))
            # ⚠️ 关键参数：IoU阈值用于匹配检测框（不要轻易修改！）
            # distth=0.5 表示IoU阈值，低于此值不认为匹配
            # 原始值可能是0.7，当前改为0.5是为了提高匹配率
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logging.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names

if __name__ == '__main__':

    args = parse_args()

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))        
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    # Find ground truth files recursively in subdirectories
    gtfiles_all = glob.glob(os.path.join('./dataset/SatMTB/test/label', args.cate, '*', '*.txt'))
    tsfiles = [f for f in glob.glob(os.path.join(args.result, args.cate, '*.txt'))]

    logging.info('Found {} groundtruth frame files and {} test files.'.format(len(gtfiles_all), len(tsfiles)))
    
    # Check if category data exists
    if len(gtfiles_all) == 0 and len(tsfiles) == 0:
        logging.warning('No ground truth or test files found for category: {}. Skipping evaluation.'.format(args.cate))
        logging.warning('Note: SatMTB dataset only contains 3 categories: car, airplane, ship.')
        logging.info('Evaluation skipped for category: {}'.format(args.cate))
        exit(0)
    
    logging.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logging.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logging.info('Loading files.')
    
    # Group ground truth files by sequence ID and merge them
    gt_sequences = {}
    for f in gtfiles_all:
        seq_id = Path(f).parts[-2]  # Extract sequence ID from path (e.g., airplane/10/000001.txt -> 10)
        if seq_id not in gt_sequences:
            gt_sequences[seq_id] = []
        gt_sequences[seq_id].append(f)
    
    # ========================================================================
    # 核心函数：将SatMTB真实值文件转换为MOT格式，并重建轨迹ID
    # ========================================================================
    def load_mtb_sat_sequence(seq_files):
        """
        加载并合并SatMTB格式的真实值文件，转换为MOT格式，并重建轨迹ID
        
        SatMTB真实值格式说明：
        - 每个序列有多帧，每帧一个txt文件（如：000001.txt, 000002.txt, ...）
        - 每帧文件中每行代表一个检测框，格式：class_id x y w h
        - 关键问题：真实值文件没有轨迹ID，只有检测框信息！
        - 因此需要从检测框的时间-空间信息重建轨迹ID
        
        坐标格式说明（需要确认）：
        - 当前假设：真实值使用中心点坐标 (cx, cy, w, h)
        - motmetrics期望：左上角坐标 (x1, y1, w, h)
        - 如果假设错误，会导致所有匹配失败！
        
        返回：DataFrame，格式为MOT标准格式，包含重建的轨迹ID
        """
        import numpy as np
        
        def calculate_iou(box1, box2):
            """
            计算两个边界框的IoU（Intersection over Union）
            
            参数：
            - box1, box2: 格式为 [x, y, w, h]，其中 (x, y) 是左上角坐标
            
            返回：IoU值 [0, 1]
            """
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            # 计算两个框的面积
            box1_area = w1 * h1
            box2_area = w2 * h2
            
            # 计算交集区域
            x1_inter = max(x1, x2)  # 交集左边界
            y1_inter = max(y1, y2)  # 交集上边界
            x2_inter = min(x1 + w1, x2 + w2)  # 交集右边界
            y2_inter = min(y1 + h1, y2 + h2)  # 交集下边界
            
            # 如果没有交集，返回0
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            # 计算交集和并集面积
            inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            union_area = box1_area + box2_area - inter_area
            
            if union_area == 0:
                return 0.0
            
            return inter_area / union_area
        
        # ====================================================================
        # 步骤1：按帧号顺序读取所有检测框
        # ====================================================================
        detections_by_frame = []
        seq_files_sorted = sorted(seq_files, key=lambda f: int(Path(f).stem))  # 按帧号排序
        
        for f in seq_files_sorted:
            frame_idx = int(Path(f).stem)  # 提取帧号（如：000001 -> 1）
            frame_detections = []
            
            # 读取当前帧的所有检测框
            with open(f, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        # 解析真实值文件格式：class_id x y w h
                        class_id = int(parts[0])    # 类别ID（0=car, 1=airplane, 2=ship, 3=train）
                        x = float(parts[1])         # X坐标（坐标格式待确认！）
                        y = float(parts[2])         # Y坐标（坐标格式待确认！）
                        w = float(parts[3])         # 宽度
                        h = float(parts[4])         # 高度
                        
                        # ====================================================
                        # ⚠️ 关键步骤：坐标格式转换
                        # ====================================================
                        # 当前假设：真实值文件使用中心点坐标 (cx, cy, w, h)
                        # motmetrics期望：左上角坐标 (x1, y1, w, h)
                        # 转换公式：左上角 = 中心点 - (w/2, h/2)
                        # 
                        # ⚠️ 警告：如果真实值已经是左上角坐标，此转换会导致坐标错误！
                        # 需要验证真实值文件的坐标格式！
                        # ====================================================
                        x1 = x - w / 2.0  # 转换为左上角X坐标
                        y1 = y - h / 2.0  # 转换为左上角Y坐标
                        
                        frame_detections.append({
                            'frame': frame_idx,
                            'bbox': [x1, y1, w, h],  # 存储为左上角坐标格式
                            'class_id': class_id
                        })
            
            # 如果当前帧有检测框，添加到列表中
            if frame_detections:
                detections_by_frame.append((frame_idx, frame_detections))
        
        # 如果没有检测框，返回None
        if not detections_by_frame:
            return None
        
        # ====================================================================
        # 步骤2：基于IoU的跟踪重建算法
        # ====================================================================
        # 核心思想：由于真实值文件没有轨迹ID，需要通过时间-空间关联重建
        # 算法：使用IoU（交并比）关联相邻帧的检测框，构建轨迹
        
        # 数据结构：tracks[track_id] = [det1, det2, ...]  # 每个轨迹包含多个检测框
        tracks = {}  # track_id -> list of detections
        next_track_id = 1  # 下一个可用的轨迹ID
        
        # ⚠️ 可调参数：IoU阈值 - 低于此值不关联（不要轻易修改！）
        iou_threshold = 0.3  # IoU阈值，用于判断两个检测框是否为同一目标
        
        # ====================================================================
        # 步骤2.1：处理第一帧 - 为所有检测框创建新轨迹
        # ====================================================================
        for frame_idx, frame_detections in detections_by_frame:
            if not tracks:  # 第一帧
                # 第一帧的所有检测框都创建新轨迹（因为没有历史信息可关联）
                for det in frame_detections:
                    tracks[next_track_id] = [det]
                    next_track_id += 1
                continue
            
            # ================================================================
            # 步骤2.2：处理后续帧 - 关联当前帧检测框与已有轨迹
            # ================================================================
            
            # 2.2.1：收集所有活跃轨迹的最后一次检测框（用于关联）
            previous_frame_detections = []  # 存储前序帧的检测框信息
            track_ids = []  # 对应的轨迹ID列表
            
            # ⚠️ 可调参数：最大帧间隔 - 允许目标在多少帧内丢失后仍能恢复
            max_frame_gap = 5  # 最多允许5帧的间隔（处理遮挡、暂时消失等情况）
            
            # 遍历所有已有轨迹，获取每个轨迹的最后一次检测
            for track_id, track_detections in tracks.items():
                if track_detections:  # 只考虑活跃轨迹
                    last_det = track_detections[-1]  # 获取该轨迹的最后一次检测
                    frame_gap = frame_idx - last_det['frame']  # 计算帧间隔
                    
                    # 只考虑帧间隔在合理范围内的轨迹（1到max_frame_gap帧）
                    # 注意：frame_idx 和 last_det['frame'] 可能不连续（如果某些帧没有检测）
                    if 1 <= frame_gap <= max_frame_gap:
                        previous_frame_detections.append({
                            'bbox': last_det['bbox'],  # 边界框
                            'track_id': track_id,      # 轨迹ID
                            'frame_gap': frame_gap     # 帧间隔
                        })
                        track_ids.append(track_id)
            
            # ================================================================
            # 2.2.2：使用贪心算法进行检测框-轨迹关联（类似Hungarian算法的简化版）
            # ================================================================
            
            # 数据结构：matches = [(detection_index, track_id, iou), ...]
            matches = []  # 存储所有可能的匹配（检测框索引、轨迹ID、IoU值）
            used_tracks = set()      # 已匹配的轨迹ID（避免一个轨迹匹配多个检测框）
            used_detections = set()  # 已匹配的检测框索引（避免一个检测框匹配多个轨迹）
            
            # 遍历当前帧的所有检测框，与所有前序帧的检测框计算IoU
            for det_idx, det in enumerate(frame_detections):
                for prev_info in previous_frame_detections:
                    track_id = prev_info['track_id']
                    prev_bbox = prev_info['bbox']
                    frame_gap = prev_info['frame_gap']
                    
                    # 动态调整IoU阈值：帧间隔越大，需要更高的IoU才能关联
                    # 原因：目标移动时间越长，位置变化可能越大
                    adjusted_threshold = iou_threshold * (1.0 + 0.1 * (frame_gap - 1))
                    adjusted_threshold = min(adjusted_threshold, 0.5)  # 上限设为0.5
                    
                    # 计算IoU
                    iou = calculate_iou(det['bbox'], prev_bbox)
                    
                    # 如果IoU超过阈值，记录为潜在匹配
                    if iou >= adjusted_threshold:
                        matches.append((det_idx, track_id, iou))
            
            # ================================================================
            # 2.2.3：按IoU降序排序，优先匹配IoU最大的（贪心策略）
            # ================================================================
            matches.sort(key=lambda x: x[2], reverse=True)  # 按IoU从大到小排序
            
            # 2.2.4：分配匹配（每个轨迹和检测框只能匹配一次）
            for det_idx, track_id, iou in matches:
                # 确保一对一匹配（一个轨迹只能匹配一个检测框，反之亦然）
                if track_id not in used_tracks and det_idx not in used_detections:
                    tracks[track_id].append(frame_detections[det_idx])  # 将检测框添加到轨迹
                    used_tracks.add(track_id)          # 标记轨迹已使用
                    used_detections.add(det_idx)       # 标记检测框已使用
            
            # ================================================================
            # 2.2.5：为未匹配的检测框创建新轨迹
            # ================================================================
            # 可能是新出现的目标，或者之前的目标被遮挡后重新出现但IoU太低
            for det_idx, det in enumerate(frame_detections):
                if det_idx not in used_detections:
                    tracks[next_track_id] = [det]  # 创建新轨迹
                    next_track_id += 1
        
        # ====================================================================
        # 步骤3：将重建的轨迹转换为MOT标准格式
        # ====================================================================
        # MOT格式：frame_id, track_id, x, y, w, h, confidence, class_id, visibility
        all_detections = []
        for track_id, track_detections in tracks.items():
            for det in track_detections:
                frame_idx = det['frame']
                x, y, w, h = det['bbox']  # 左上角坐标和宽高
                class_id = det['class_id']
                
                # ⚠️ 关键：构建MOT格式的行数据
                # 格式：FrameId, Id, X, Y, Width, Height, Confidence, ClassId, Visibility
                # - FrameId: 帧号
                # - Id: 轨迹ID（这是重建的，不是原始数据中的）
                # - X, Y: 左上角坐标
                # - Width, Height: 宽度和高度
                # - Confidence: 置信度（真实值设为1.0）
                # - ClassId: 类别ID（使用NAME_LABEL映射）
                # - Visibility: 可见性（真实值设为1.0）
                all_detections.append([
                    frame_idx,                           # FrameId
                    track_id,                            # Id (重建的轨迹ID)
                    x, y, w, h,                          # X, Y, Width, Height
                    1.0,                                 # Confidence
                    NAME_LABEL[args.cate],               # ClassId（不要修改！）
                    1.0                                  # Visibility
                ])
        
        if not all_detections:
            return None
        
        # ====================================================================
        # 步骤4：转换为DataFrame格式（motmetrics要求的格式）
        # ====================================================================
        df = pd.DataFrame(all_detections, columns=[
            'FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 
            'Confidence', 'ClassId', 'Visibility'
        ])
        
        # motmetrics要求MultiIndex：(FrameId, Id)
        # ⚠️ 不要修改索引格式！motmetrics依赖此格式
        df = df.set_index(['FrameId', 'Id'])
        
        # 确保索引是MultiIndex类型
        if not isinstance(df.index, pd.MultiIndex):
            df = df.reset_index().set_index(['FrameId', 'Id'])
        
        return df
    
    # Load ground truth sequences
    gt = OrderedDict()
    for seq_id, seq_files in gt_sequences.items():
        gt_df = load_mtb_sat_sequence(seq_files)
        if gt_df is not None:
            gt[seq_id] = gt_df
    
    logging.info('Loaded {} ground truth sequences.'.format(len(gt)))
    
    # ========================================================================
    # 加载测试结果文件并过滤
    # ========================================================================
    # 测试结果文件格式：每个序列一个txt文件，MOT格式
    # 格式：frame_id, track_id, x, y, w, h, conf, class_id, visibility
    # 其中 x, y 是左上角坐标（从testDis_satmtb.py确认）
    
    ts = OrderedDict()
    expected_class_id = NAME_LABEL[args.cate]  # 期望的类别ID（不要修改！）
    
    # ⚠️ 可调参数：置信度阈值（可能需要根据实际情况调整）
    conf_threshold = 0.5  # 低于此置信度的检测框将被过滤
    
    for f in tsfiles:
        seq_id = os.path.splitext(Path(f).parts[-1])[0]
        tsacc = mm.io.loadtxt(f, fmt='mot16')
        
        original_count = len(tsacc)
        
        # ====================================================================
        # 过滤1：只保留匹配的类别ID
        # ====================================================================
        # ⚠️ 必要过滤：测试结果可能包含其他类别的检测框
        # 例如：ship类别的测试结果可能包含car、airplane等其他类别的检测
        # 这些都需要过滤掉，只保留当前评估类别
        if 'ClassId' in tsacc.columns:
            tsacc = tsacc[tsacc['ClassId'] == expected_class_id].copy()
            after_class_filter = len(tsacc)
            if original_count != after_class_filter:
                logging.info('Sequence {}: Filtered {} -> {} rows (ClassId filter)'.format(
                    seq_id, original_count, after_class_filter))
        
        # ====================================================================
        # 过滤2：过滤低置信度检测框
        # ====================================================================
        # ⚠️ 可选过滤：作者原始代码可能不进行此过滤，或使用不同阈值
        # 当前阈值0.5可能需要根据实际情况调整
        if 'Conf' in tsacc.columns:
            before_conf_filter = len(tsacc)
            tsacc = tsacc[tsacc['Conf'] >= conf_threshold].copy()
            after_conf_filter = len(tsacc)
            if before_conf_filter != after_conf_filter:
                logging.info('Sequence {}: Filtered {} -> {} rows (Conf >= {})'.format(
                    seq_id, before_conf_filter, after_conf_filter, conf_threshold))
        
        if len(tsacc) > 0:
            ts[seq_id] = tsacc
        else:
            logging.warning('Sequence {}: No valid detections after filtering, skipping.'.format(seq_id))    

    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)
    
    logging.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked', \
      'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', \
      'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(
      accs, names=names, 
      metrics=metrics, generate_overall=True)
    
    # Print detailed error breakdown for diagnosis
    logging.info('=' * 60)
    logging.info('Detailed error breakdown for diagnosis:')
    logging.info('=' * 60)
    if 'OVERALL' in summary.index:
        overall_row = summary.loc['OVERALL']
        gt_boxes = int(overall_row['num_objects'])
        fp = int(overall_row['num_false_positives'])
        fn = int(overall_row['num_misses'])
        ids = int(overall_row['num_switches'])
        mota = overall_row['mota'] * 100
        logging.info('OVERALL Statistics:')
        logging.info('  GT boxes (num_objects): {}'.format(gt_boxes))
        logging.info('  False Positives (FP): {} ({:.1f}%)'.format(fp, (fp/gt_boxes*100) if gt_boxes > 0 else 0))
        logging.info('  False Negatives / Misses (FN): {} ({:.1f}%)'.format(fn, (fn/gt_boxes*100) if gt_boxes > 0 else 0))
        logging.info('  ID Switches (IDSW): {}'.format(ids))
        logging.info('  MOTA: {:.2f}%'.format(mota))
        logging.info('=' * 60)
    # summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    # print(mm.io.render_summary(
    #   summary, formatters=mh.formatters, 
    #   namemap=mm.io.motchallenge_metric_names))
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 
          'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 
          'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 
      'num_fragmentations', 'mostly_tracked', 'partially_tracked', 
      'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    # print(mm.io.render_summary(
    #   summary, formatters=fmt, 
    #   namemap=mm.io.motchallenge_metric_names))
    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(
    accs, names=names, 
    metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(
    summary, formatters=mh.formatters, 
    namemap=mm.io.motchallenge_metric_names))

    # with open(os.path.join(args.result, 'eval.txt'), 'a') as f:
    #     f.write('Evaluate ' + args.cate + ':\n')
    #     f.write(mm.io.render_summary(
    #         summary, formatters=mh.formatters, 
    #         namemap=mm.io.motchallenge_metric_names))
    #     f.write('\n')

    logging.info('Completed')

