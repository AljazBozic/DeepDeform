import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import cv2
import utils
import numpy as np
import config as cfg
import flow_vis


def main():
    dataset_dir = cfg.DATA_ROOT_DIR
    alignments_path = os.path.join(dataset_dir, "{0}_selfsupervised.json".format(cfg.DATA_TYPE))
    flow_normalization = 100.0; # in pixels

    with open(alignments_path, "r") as f:
        pairs = json.load(f)


    for pair in pairs:
        source_color = cv2.imread(os.path.join(dataset_dir, pair["source_color"]))
        target_color = cv2.imread(os.path.join(dataset_dir, pair["target_color"]))
        optical_flow = utils.load_flow(os.path.join(dataset_dir, pair["optical_flow"]))
        
        optical_flow = np.moveaxis(optical_flow, 0, -1) # (h, w, 2)

        invalid_flow = optical_flow == -np.Inf
        optical_flow[invalid_flow] = 0.0

        flow_color = flow_vis.flow_to_color(optical_flow, convert_to_bgr=False)

        cv2.imshow("Source", source_color)
        cv2.imshow("Target", target_color)
        cv2.imshow("Flow", flow_color)

        cv2.waitKey(0)


if __name__ == "__main__":
    main()