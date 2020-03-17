import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import cv2
import utils
import numpy as np
import random
import config as cfg


def main():
    dataset_dir = cfg.DATA_ROOT_DIR
    matches_path = os.path.join(dataset_dir, "{0}_matches.json".format(cfg.DATA_TYPE))

    with open(matches_path, "r") as f:
        matches = json.load(f)

    for pair in matches:
        source_color = cv2.imread(os.path.join(dataset_dir, pair["source_color"]))
        target_color = cv2.imread(os.path.join(dataset_dir, pair["target_color"]))

        h, w = source_color.shape[:2]
        match_vis = np.zeros((h, 2 * w, 3), np.uint8)
        match_vis[:h, :w, :] = source_color  
        match_vis[:h, w:, :] = target_color
        match_vis[:, :, 1] = match_vis[:, :, 0]  
        match_vis[:, :, 2] = match_vis[:, :, 0]

        for match in pair["matches"]:
            color = tuple([random.randint(0, 255) for _ in range(3)])
            thickness = 2
            cv2.line(match_vis, (int(match["source_x"]), int(match["source_y"])) , (int(match["target_x"] + w), int(match["target_y"])), color, thickness)

        cv2.imshow("Source", source_color)
        cv2.imshow("Target", target_color)
        cv2.imshow("Matches", match_vis)

        cv2.waitKey(0)


if __name__ == "__main__":
    main()