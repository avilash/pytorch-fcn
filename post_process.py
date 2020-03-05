import _init_paths
import os
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.gen_utils import make_dir_if_not_exist

from config.base_config import cfg, cfg_from_file


def post_process(exp_name):
    valid_ext = [".jpg", ".png"]
    all_file_names = []

    src_path = os.path.join('data', exp_name, 'results')
    base_path = os.path.join('data', exp_name, 'pp_results')
    pp_img_path = os.path.join(base_path, 'imgs')
    pp_preds_path = os.path.join(base_path, 'preds')
    pp_wt_imgs_path = os.path.join(base_path, 'wt_imgs')

    make_dir_if_not_exist(pp_img_path)
    make_dir_if_not_exist(pp_preds_path)
    make_dir_if_not_exist(pp_wt_imgs_path)

    for f in os.listdir(os.path.join(src_path, 'imgs')):
        file_ext = os.path.splitext(f)[1]
        file_name = os.path.splitext(f)[0]
        if file_ext.lower() not in valid_ext:
            continue

        img = cv2.imread(os.path.join(src_path, 'imgs', f))
        pred = cv2.imread(os.path.join(src_path, 'preds', file_name + '_pred.png'), cv2.IMREAD_GRAYSCALE)

        pp_pred = np.zeros_like(pred)

        rho = 1
        theta = np.pi / 180
        threshold = 28
        min_line_len = 40
        max_line_gap = 5

        lines = cv2.HoughLinesP(pred, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)

        if lines is not None:
            for x in range(0, len(lines)):
                for x1, y1, x2, y2 in lines[x]:
                    cv2.line(pp_pred, (x1, y1), (x2, y2), 255, 2)

        red_mask = np.zeros_like(img)
        red_mask[pp_pred == 255] = (0, 0, 255)
        weighted_img = cv2.addWeighted(red_mask, 0.4, img, 1, 0)

        cv2.imwrite(os.path.join(pp_img_path, f), img)
        cv2.imwrite(os.path.join(pp_wt_imgs_path, f), weighted_img)
        cv2.imwrite(os.path.join(pp_preds_path, file_name + '_pred.png'), pp_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Road Lanes Post Process')
    parser.add_argument('--exp_name', default='exp0', type=str,
                        help='name of experiment')
    args = parser.parse_args()

    post_process(args.exp_name)
