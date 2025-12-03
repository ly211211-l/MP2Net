"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.
Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
Modified by Xingyi Zhou
"""

import argparse
import glob
import os
import logging
import numpy as np
# Fix NumPy 2.0 compatibility: add back np.asfarray if missing
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=None: np.asarray(a, dtype=dtype if dtype else np.float64)
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
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.7))
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

    # Use original code's file path logic: ./data/MTB-SAT/test/label/{cate}/*.txt
    gtfiles = glob.glob(os.path.join('./data/MTB-SAT/test/label', args.cate, '*.txt'))
    tsfiles = [f for f in glob.glob(os.path.join(args.result, args.cate, '*.txt'))]

    logging.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logging.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logging.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logging.info('Loading files.')
    
    # Load GT: Original code uses fmt='mtb-sat' but motmetrics doesn't support it
    # The files are in MOT format with 11 fields: frame_id,track_id,x,y,w,h,conf,class_id,visibility,?,?
    # We need to manually parse to correctly extract all fields
    gt = OrderedDict()
    for f in gtfiles:
        seq_id = os.path.splitext(Path(f).parts[-1])[0]
        try:
            all_detections = []
            with open(f, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if len(parts) >= 9:
                        frame_id = int(float(parts[0]))
                        track_id = int(float(parts[1]))
                        x = float(parts[2])
                        y = float(parts[3])
                        w = float(parts[4])
                        h = float(parts[5])
                        conf = float(parts[6])
                        class_id = int(float(parts[7]))
                        visibility = float(parts[8])
                        
                        # Filter by class_id (original code's intent via NAME_LABEL parameter)
                        if class_id == NAME_LABEL[args.cate]:
                            all_detections.append([frame_id, track_id, x, y, w, h, conf, class_id, visibility])
            
            if all_detections:
                df = pd.DataFrame(all_detections, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'])
                df = df.set_index(['FrameId', 'Id'])
                gt[seq_id] = df
        except Exception as e:
            logging.warning('Failed to load GT file {}: {}'.format(f, e))
    
    # Load test results: Original code uses fmt='mot16' with NAME_LABEL as positional parameter
    # Fixed: Use objid as keyword parameter to avoid parameter conflict
    # Note: objid parameter may not work correctly, so we manually filter by ClassId
    ts = OrderedDict()
    for f in tsfiles:
        seq_id = os.path.splitext(Path(f).parts[-1])[0]
        try:
            df = mm.io.loadtxt(f, fmt='mot16')
            # Filter by ClassId to only keep the target category
            if 'ClassId' in df.columns:
                df = df[df['ClassId'] == NAME_LABEL[args.cate]]
            ts[seq_id] = df
        except Exception as e:
            logging.warning('Failed to load test file {}: {}'.format(f, e))
    
    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)
    
    logging.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked', \
      'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', \
      'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(
      accs, names=names, 
      metrics=metrics, generate_overall=True)
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

