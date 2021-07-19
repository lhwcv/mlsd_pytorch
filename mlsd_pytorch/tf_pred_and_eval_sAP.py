import os
import cv2
import json
import tqdm
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf


def msTPFP(line_pred, line_gt, threshold):
    line_pred = line_pred.reshape(-1, 2, 2)[:, :, ::-1]
    line_gt = line_gt.reshape(-1, 2, 2)[:, :, ::-1]
    diff = ((line_pred[:, None, :, None] - line_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )

    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    hit = np.zeros(len(line_gt), np.bool)
    tp = np.zeros(len(line_pred), np.float)
    fp = np.zeros(len(line_pred), np.float)
    for i in range(len(line_pred)):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def TPFP(lines_dt, lines_gt, threshold):
    lines_dt = lines_dt.reshape(-1,2,2)[:,:,::-1]
    lines_gt = lines_gt.reshape(-1,2,2)[:,:,::-1]
    diff = ((lines_dt[:, None, :, None] - lines_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )
    choice = np.argmin(diff,1)
    dist = np.min(diff,1)
    hit = np.zeros(len(lines_gt), np.bool)
    tp = np.zeros(len(lines_dt), np.float)
    fp = np.zeros(len(lines_dt),np.float)

    for i in range(lines_dt.shape[0]):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp

def AP(tp, fp):
    recall = tp
    precision = tp/np.maximum(tp+fp, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]

    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])

    return ap

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--gt_json", type=str, default="/home/lhw/data/wireframe_parsing/wireframe_afm/test.json")
    args.add_argument("--img_dir", type=str, default="/home/lhw/data/wireframe_parsing/wireframe_afm/images/")
    args.add_argument("--thresh", type=float, help="sAP thresh", default=10.0)
    return args.parse_args()


model_name = 'tflite_models/M-LSD_512_tiny_fp32.tflite'
interpreter = tf.lite.Interpreter(model_path=model_name)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



def pred_lines_fn(image, interpreter, input_details, output_details, input_shape=[512, 512], score_thr=0.01, dist_thr=2.0):
    h, w, _ = image.shape
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]

    resized_image = np.concatenate([cv2.resize(image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA), np.ones([input_shape[0], input_shape[1], 1])], axis=-1)
    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')
    interpreter.set_tensor(input_details[0]['index'], batch_image)
    interpreter.invoke()

    pts = interpreter.get_tensor(output_details[0]['index'])[0]
    pts_score = interpreter.get_tensor(output_details[1]['index'])[0]
    vmap = interpreter.get_tensor(output_details[2]['index'])[0]

    start = vmap[:,:,:2]
    end = vmap[:,:,2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    segments_list = []
    scores_list = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])
            scores_list.append(score)
    
    lines = 2 * np.array(segments_list) # 256 > 512
    lines[:,0] = lines[:,0] * w_ratio
    lines[:,1] = lines[:,1] * h_ratio
    lines[:,2] = lines[:,2] * w_ratio
    lines[:,3] = lines[:,3] * h_ratio

    return lines,scores_list


def infer_one(img_fn, input_size=512, score_thresh=0.0, min_len=0.0, topk=200):
    img = cv2.imread(img_fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    img = cv2.resize(img, (input_size, input_size))
    
    pred_lines, scores = pred_lines_fn(img, interpreter, input_details, output_details, 
                                       input_shape=[input_size, input_size], score_thr=score_thresh, dist_thr=min_len)

    pred_lines_list = []
    scores_list = []
    for line, score in zip(pred_lines, scores):
        x0, y0, x1, y1 = line

        x0 = w * x0 / input_size
        x1 = w * x1 / input_size

        y0 = h * y0 / input_size
        y1 = h * y1 / input_size

        pred_lines_list.append([x0, y0, x1, y1])
        scores_list.append(score)

    return {
        'full_fn': img_fn,
        'filename': os.path.basename(img_fn),
        'width': w,
        'height': h,
        'lines': pred_lines_list,
        'scores': scores_list
    }


def calculate_sAP(gt_infos, pred_infos, sap_thresh):
    assert len(gt_infos) == len(pred_infos)

    tp_list, fp_list, scores_list = [], [], []
    n_gt = 0

    for gt, pred in zip(gt_infos, pred_infos):
        assert gt['filename'] == pred['filename']
        h, w = gt['height'], gt['width']
        pred_lines = np.array(pred['lines'], np.float32)
        pred_scores = np.array(pred['scores'], np.float32)

        gt_lines = np.array(gt['lines'], np.float32)
        scale = np.array([128.0/ w, 128.0/h, 128.0/ w, 128.0/h], np.float32)
        pred_lines_128 = pred_lines * scale
        gt_lines_128 = gt_lines * scale

        tp, fp = msTPFP(pred_lines_128, gt_lines_128, sap_thresh)

        n_gt += gt_lines_128.shape[0]
        tp_list.append(tp)
        fp_list.append(fp)
        scores_list.append(pred_scores)

    tp_list = np.concatenate(tp_list)
    fp_list = np.concatenate(fp_list)
    scores_list = np.concatenate(scores_list)
    idx = np.argsort(scores_list)[::-1]
    tp = np.cumsum(tp_list[idx]) / n_gt
    fp = np.cumsum(fp_list[idx]) / n_gt
    rcs = tp
    pcs = tp / np.maximum(tp + fp, 1e-9)
    sAP = AP(tp, fp) * 100

    return  sAP


def main(args):
    
    label_file = args.gt_json
    img_dir = args.img_dir
    contens = json.load(open(label_file, 'r'))

    gt_infos = []
    pred_infos = []

    for c in tqdm.tqdm(contens):
        gt_infos.append(c)
        fn = c['filename'][:-4] + '.png'
        full_fn = img_dir + '/' + fn
        pred_infos.append(infer_one(full_fn))

    ap = calculate_sAP(gt_infos, pred_infos, args.thresh)

    print("====> sAP{}: {}".format(args.thresh, ap))


if __name__ == '__main__':
    main(get_args())
    

