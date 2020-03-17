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
    occlusions_path = os.path.join(dataset_dir, "{0}_occlusions.json".format(cfg.DATA_TYPE))

    with open(occlusions_path, "r") as f:
        occlusions = json.load(f)

    for pair in occlusions:
        source_color = cv2.imread(os.path.join(dataset_dir, pair["source_color"]))
        target_color = cv2.imread(os.path.join(dataset_dir, pair["target_color"]))

        h, w = source_color.shape[:2]
        occlusion_vis = np.copy(source_color)
        occlusion_vis[:, :, 1] = occlusion_vis[:, :, 0]  
        occlusion_vis[:, :, 2] = occlusion_vis[:, :, 0]

        for occlusion in pair["occlusions"]:
            color = tuple([random.randint(0, 255) for _ in range(3)])
            radius = 5
            thickness = -1
            cv2.circle(occlusion_vis, (int(occlusion["source_x"]), int(occlusion["source_y"])), radius, color, thickness)

        cv2.imshow("Source", source_color)
        cv2.imshow("Target", target_color)
        cv2.imshow("Occlusions", occlusion_vis)

        cv2.waitKey(0)


if __name__ == "__main__":
    main()