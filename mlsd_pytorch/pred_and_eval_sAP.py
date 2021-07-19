import  sys
import os
sys.path.append(os.path.dirname(__file__) +'/../')
import cv2
import torch
import json
import tqdm
import argparse
import numpy as np
from mlsd_pytorch.cfg.default import get_cfg_defaults
from mlsd_pytorch.models.build_model import build_model
from mlsd_pytorch.data.utils import deccode_lines
from mlsd_pytorch.metric import msTPFP, AP
from albumentations import Normalize

def get_args():
    args = argparse.ArgumentParser()
    current_dir = default=os.path.dirname(__file__)
    args.add_argument("--config", type=str,default = current_dir  + '/configs/mobilev2_mlsd_tiny_512_base2_bsize24.yaml')
    args.add_argument("--model_path", type=str,
                      default= current_dir +"/../workdir/pretrained_models/mobilev2_mlsd_tiny_512_bsize24/best.pth")
    args.add_argument("--gt_json", type=str,
                      default= current_dir +"/../data/wireframe_raw/valid.json")
    args.add_argument("--img_dir", type=str,
                      default= current_dir + "/../data/wireframe_raw/images/")
    args.add_argument("--sap_thresh", type=float, help="sAP thresh", default=10.0)
    args.add_argument("--top_k", type=float, help="top k lines", default= 500)
    args.add_argument("--min_len", type=float, help="min len of line", default=5.0)
    args.add_argument("--score_thresh", type=float, help="line score thresh", default=0.05)
    args.add_argument("--input_size", type=int, help="image input size", default=512)

    return args.parse_args()

test_aug = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

def infer_one(img_fn, model, input_size=512, score_thresh=0.01, min_len=0, topk=200):
    img = cv2.imread(img_fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    img = cv2.resize(img, (input_size, input_size))
    #img = (img / 127.5) - 1.0
    img = test_aug(image=img)['image']
    img = img.transpose(2,0,1)
    img = torch.from_numpy(img).unsqueeze(0).float().cuda()

    with torch.no_grad():
        batch_outputs = model(img)
    tp_mask = batch_outputs[:, 7:, :, :]

    center_ptss, pred_lines, scores = deccode_lines(tp_mask, score_thresh, min_len, topk, 3)

    pred_lines = pred_lines.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()

    pred_lines_list = []
    scores_list = []
    for line, score in zip(pred_lines, scores):
        x0, y0, x1, y1 = line

        x0 = w * x0 / (input_size / 2)
        x1 = w * x1 / (input_size / 2)

        y0 = h * y0 / (input_size / 2)
        y1 = h * y1 / (input_size / 2)

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
    cfg = get_cfg_defaults()
    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    print('using config: ', args.config.strip())
    cfg.merge_from_file(args.config)

    model = build_model(cfg).cuda().eval()
    model.load_state_dict(torch.load(args.model_path), strict=True)

    label_file = args.gt_json
    img_dir = args.img_dir
    contens = json.load(open(label_file, 'r'))

    gt_infos = []
    pred_infos = []

    for c in tqdm.tqdm(contens):
        gt_infos.append(c)
        fn = c['filename'][:-4] + '.png'
        full_fn = img_dir + '/' + fn
        pred_infos.append(infer_one(full_fn, model,
                                    args.input_size,
                                    args.score_thresh,
                                    args.min_len, args.top_k ))

    ap = calculate_sAP(gt_infos, pred_infos, args.sap_thresh)

    print("====> sAP{}: {}".format(args.sap_thresh, ap))


if __name__ == '__main__':
    main(get_args())
