import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pathlib
import subprocess
import os
import shutil
import operator
import glob
import csv
import re
import json
import utils
from enum import Enum
from tqdm import tqdm
import config as cfg


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


class Status(Enum):
    SUCCESS = 0
    FLOW_NOT_FOUND = 1
    INVALID_FLOW_DIMENSIONS = 2


def in_bounds(pixel, width, height):
    return pixel[0] >= 0 and pixel[0] < width and pixel[1] >= 0 and pixel[1] < height


def evaluate(groundtruth_matches_json_path, predictions_dir):
    print("#"*66)
    print("FLOW EVALUATION")
    print("#"*66)

    # Load groundtruth matches.
    with open(groundtruth_matches_json_path, 'r') as f:
        matches = json.load(f)

    # Compute pixel distances for every annotated keypoint,
    # for every sequence.
    pixel_dist_sum_per_seq = {}
    valid_pixel_num_per_seq = {}
    accurate_pixel_num_per_seq = {}

    pixel_threshold = 20.0

    print()
    print("Groundtruth matches: {}".format(groundtruth_matches_json_path))
    print("Predictions: {}".format(predictions_dir))
    print()

    for frame_pair in tqdm(matches):
        seq_id = frame_pair["seq_id"]
        object_id = frame_pair["object_id"]
        source_id = frame_pair["source_id"]
        target_id = frame_pair["target_id"]

        flow_id = "{0}_{1}_{2}_{3}.oflow".format(seq_id, object_id, source_id, target_id)
        flow_pred_path = os.path.join(predictions_dir, flow_id)
        if not os.path.exists(flow_pred_path):
            print("Flow prediction missing: {}".format(flow_pred_path))
            return { "status": Status.FLOW_NOT_FOUND }

        flow_image_pred = utils.load_flow(flow_pred_path)
        if flow_image_pred.shape[1] != IMAGE_HEIGHT or flow_image_pred.shape[2] != IMAGE_WIDTH:
            print("Invalid flow dimesions:", flow_image_pred.shape)
            return { "status": Status.INVALID_FLOW_DIMENSIONS }

        flow_image_pred = np.moveaxis(flow_image_pred, 0, -1)

        for match in frame_pair["matches"]:
            # Read keypoint and match.
            source_kp = np.array([match["source_x"], match["source_y"]])
            target_kp = np.array([match["target_x"], match["target_y"]])

            source_kp_rounded = np.round(source_kp)
            target_kp_rounded = np.round(target_kp)

            # Make sure it's in bounds.
            assert in_bounds(source_kp_rounded, IMAGE_WIDTH, IMAGE_HEIGHT) and \
                    in_bounds(source_kp_rounded, IMAGE_WIDTH, IMAGE_HEIGHT)

            source_v, source_u = source_kp_rounded[1].astype(np.int64), source_kp_rounded[0].astype(np.int64)

            flow_pred = flow_image_pred[source_v, source_u]
            flow_gt = (target_kp - source_kp_rounded).astype(np.float32)

            diff = flow_pred - flow_gt
            pixel_dist = np.sum(diff * diff)
            if pixel_dist > 0: pixel_dist = np.sqrt(pixel_dist)
            pixel_dist = float(pixel_dist)

            pixel_dist_sum_per_seq[seq_id] = pixel_dist_sum_per_seq.get(seq_id, 0.0) + pixel_dist
            valid_pixel_num_per_seq[seq_id] = valid_pixel_num_per_seq.get(seq_id, 0) + 1
            
            if pixel_dist <= pixel_threshold:
                accurate_pixel_num_per_seq[seq_id] = accurate_pixel_num_per_seq.get(seq_id, 0) + 1

    # Compute total statistics.
    pixel_dist_sum = 0.0
    valid_pixel_num = 0
    accurate_pixel_num = 0

    print()
    print("{0:<20s} | {1:^20s} | {2:^20s}".format("Sequence ID", "Accuracy (<20px)", "EPE (pixel)"))
    print("-"*66)

    pixel_dist_per_seq = {}
    pixel_acc_per_seq = {}
    for seq_id in sorted(pixel_dist_sum_per_seq.keys()):
        pixel_dist_sum_seq = pixel_dist_sum_per_seq[seq_id]
        valid_pixel_num_seq = valid_pixel_num_per_seq[seq_id]
        accurate_pixel_num_seq = accurate_pixel_num_per_seq.get(seq_id, 0)

        pixel_dist_sum += pixel_dist_sum_seq
        valid_pixel_num += valid_pixel_num_seq
        accurate_pixel_num += accurate_pixel_num_seq

        pixel_dist_seq = pixel_dist_sum_seq / valid_pixel_num_seq if valid_pixel_num_seq > 0 else -1.0
        pixel_acc_seq = accurate_pixel_num_seq / valid_pixel_num_seq if valid_pixel_num_seq > 0 else -1.0
        
        pixel_dist_per_seq[seq_id] = pixel_dist_seq
        pixel_acc_per_seq[seq_id] = pixel_acc_seq

        print("{0:<20s} | {1:^20.4f} | {2:^20.3f}".format(seq_id, pixel_acc_seq, pixel_dist_seq))

    pixel_dist = pixel_dist_sum / valid_pixel_num if valid_pixel_num > 0 else -1.0
    pixel_acc = accurate_pixel_num / valid_pixel_num if valid_pixel_num > 0 else -1.0
    print("-"*66)
    print("{0:<20s} | {1:^20.4f} | {2:^20.3f}".format("Total", pixel_acc, pixel_dist))

    return {
        "status": Status.SUCCESS, 
        "pixel_accuracy": pixel_acc, 
        "pixel_distance": pixel_dist, 
        "pixel_accuracy_per_seq": pixel_acc_per_seq, 
        "pixel_distance_per_seq": pixel_dist_per_seq
    }


if __name__ == "__main__":
    groundtruth_matches_json_path = os.path.join(cfg.DATA_ROOT_DIR, "{0}_matches.json".format(cfg.DATA_TYPE))
    predictions_dir = cfg.FLOW_RESULTS_DIR

    evaluate(groundtruth_matches_json_path, predictions_dir)