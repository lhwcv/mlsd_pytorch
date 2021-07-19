import torch
import torch.nn as nn
import numpy as np
import cv2
from  torch.nn import functional as F

def swap_line_pt_maybe(line):
    '''
    [x0, y0, x1, y1]
    '''
    L = line
    # if line[0] > line[2]:
    #     L = [line[2], line[3], line[0], line[1]]
    if abs(line[0] - line[2]) > abs(line[1] - line[3]):
        if line[0] > line[2]:
            L = [line[2], line[3], line[0], line[1]]
    else:
        if line[1] > line[3]:
            L = [line[2], line[3], line[0], line[1]]
    return L

def deccode_output_score_and_ptss(tpMap, topk_n = 200, ksize = 5):
    '''
    tpMap:
    center: tpMap[1, 0, :, :]
    displacement: tpMap[1, 1:5, :, :]
    '''
    b, c, h, w = tpMap.shape
    assert  b==1, 'only support bsize==1'
    displacement = tpMap[:, 1:5, :, :][0]
    center = tpMap[:, 0, :, :]
    heat = torch.sigmoid(center)
    hmax = F.max_pool2d( heat, (ksize, ksize), stride=1, padding=(ksize-1)//2)
    keep = (hmax == heat).float()
    heat = heat * keep
    heat = heat.reshape(-1, )

    scores, indices = torch.topk(heat, topk_n, dim=-1, largest=True)
    yy = torch.floor_divide(indices, w).unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)
    ptss = torch.cat((yy, xx),dim=-1)

    ptss   = ptss.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    displacement = displacement.detach().cpu().numpy()

    return  ptss, scores, displacement

# def deccode_lines(tpMap,score_thr=0.1, dist_thr= 20, topk_n = 200, ksize = 3):
#     pts, pts_score, vmap = deccode_output_score_and_ptss(tpMap, topk_n=topk_n, ksize=ksize)
#
#     start = vmap[:2, :, :]
#     end = vmap[2:, :, :]
#     dist_map = np.sqrt(np.sum((start - end) ** 2, axis=0))
#
#     segments_list = []
#     scores = []
#     for center, score in zip(pts, pts_score):
#         y, x = center
#         distance = dist_map[y, x]
#         if score > score_thr and distance > dist_thr:
#             disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[:, y, x]
#             x_start = x + disp_x_start
#             y_start = y + disp_y_start
#             x_end = x + disp_x_end
#             y_end = y + disp_y_end
#             segments_list.append([x_start, y_start, x_end, y_end])
#             scores.append(score)
#
#     lines = np.array(segments_list)
#     return lines, scores

def deccode_lines(tpMap,score_thresh = 0.1, len_thresh=2, topk_n = 1000, ksize = 3 ):
    '''
    tpMap:
    center: tpMap[1, 0, :, :]
    displacement: tpMap[1, 1:5, :, :]
    '''
    b, c, h, w = tpMap.shape
    assert  b==1, 'only support bsize==1'
    displacement = tpMap[:, 1:5, :, :]
    center = tpMap[:, 0, :, :]
    heat = torch.sigmoid(center)
    hmax = F.max_pool2d( heat, (ksize, ksize), stride=1, padding=(ksize-1)//2)
    keep = (hmax == heat).float()
    heat = heat * keep
    heat = heat.reshape(-1, )
    
    heat = torch.where(heat <score_thresh, torch.zeros_like(heat), heat)

    scores, indices = torch.topk(heat, topk_n, dim=-1, largest=True)
    valid_inx = torch.where(scores > score_thresh)
    scores = scores[valid_inx]
    indices = indices[valid_inx]

    yy = torch.floor_divide(indices, w).unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)
    center_ptss = torch.cat((xx, yy),dim=-1)

    start_point = center_ptss + displacement[0, :2, yy, xx].reshape(2, -1).permute(1,0)
    end_point = center_ptss + displacement[0, 2:, yy, xx].reshape(2, -1).permute(1,0)

    lines = torch.cat((start_point, end_point), dim=-1)

    all_lens = (end_point - start_point) ** 2
    all_lens = all_lens.sum(dim=-1)
    all_lens = torch.sqrt(all_lens)
    valid_inx = torch.where(all_lens > len_thresh)

    center_ptss = center_ptss[valid_inx]
    lines = lines[valid_inx]
    scores = scores[valid_inx]

    return center_ptss, lines, scores


def _max_pool_np(x, kernel=5):
    heat = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    heat = hmax.numpy()[0]
    return heat


def _nms(heat, kernel=3):
    is_np = isinstance(heat, np.ndarray)
    if is_np:
        heat = torch.from_numpy(heat).unsqueeze(0)

    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)

    keep = (hmax == heat).float()
    heat = heat * keep

    if is_np:
        heat = heat.cpu().numpy()[0]

    return heat


def TP_map_to_line_numpy(centermap, dis_map, thresh=0.2, inputW = 512, inputH= 512):
    """
    centermap: (1, h, w)
    dis_map:   (4, h, w)
    """
    _, h, w = centermap.shape
    h_ratio, w_ratio = [h / inputH, w / inputW]

    center_nms = _nms(centermap, kernel=3)[0]
    # print('center_nms.shape:', center_nms.shape)

    center_pos = np.where(center_nms > thresh)
    ## [y, x]
    center_pos = np.array([center_pos[1], center_pos[0]])
    # print("center_pos.shape:", center_pos.shape)

    dis_list = dis_map[:, center_pos[1], center_pos[0]]
    #print(dis_list)
    ## [x, y]
    dis_list = dis_list.transpose(1, 0)

    center_pos = center_pos.transpose(1, 0)

    #cale = np.array([w / 100.0, h / 100.0])
    scale = np.array([w_ratio, h_ratio])
    start_point = center_pos + dis_list[:, 0:2] * scale * 2
    end_point = center_pos + dis_list[:, 2:4] * scale * 2

    line = np.stack([start_point, end_point], axis=1)
    return 2 *line.reshape((-1, 4))

# def work_around_line(x0, y0, x1, y1, n=2, r=0.0, thickness = 3):
#     t = (thickness - 1) // 2
#
#     if abs(x0 - x1) > abs(y0 -y1):
#         ## y = k* x + b
#         k = (y1 - y0) / (x1 - x0)
#         b = y1 - k * x1
#
#         ptss = []
#         xc = (x0 + x1) / 2
#         if n is None:
#             n = int(abs(x1 - x0) * r)
#
#         xmin = int(xc - n)
#         xmax = int(xc + n)
#         for x in range(xmin, xmax+1):
#             y = k * x + b
#             for iy in range(thickness):
#                 ptss.append([x, y + t - iy])
#
#         return  ptss
#     else:
#         ## x = k* y + b
#         k = (x1 - x0) / (y1 - y0)
#         b = x1 - k * y1
#         ptss = []
#
#         yc = (y0 + y1) / 2
#         if n is None:
#             n = int(abs(y1 - y0) * r)
#         ymin = int(yc - n)
#         ymax = int(yc + n)
#
#         for y in range(ymin, ymax+1):
#             x =k * y + b
#             for ix in range(thickness):
#                 ptss.append([x + t - ix, y])
#         return ptss


# def near_area_n(xc, yc, n= 5):
#     n = n // 2
#     ptss = []
#     for x in range(xc-n,  xc + n +1):
#         for y in range(yc - n, yc +n +1):
#             ptss.append([x, y])
#     return  ptss

# def line_len_and_angle(x0, y0, x1, y1):
#     if abs(x0 - x1) < 1e-6:
#         ang = np.pi / 2
#     else:
#         ang = np.arctan( abs ( (y0 -y1) / (x0 -x1) ) )
#
#     ang = ang / (2 * np.pi) + 0.5
#     lens = np.sqrt( (x0 - x1) **2 + (y0 - y1) **2)
#
#     return  lens, ang


def line_len_and_angle(x0, y0, x1, y1):
    if abs(x0 - x1) < 1e-6:
        ang = np.pi / 2
    else:
        ang = np.arctan(abs((y0 - y1) / (x0 - x1)))

    ang = ang / (2 * np.pi) + 0.5
    len = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    return len, ang


def near_area_n(xc, yc, n=5):
    if n <= 1:
        return [[xc, yc]]
    n = n // 2
    ptss = []
    for x in range(xc - n, xc + n + 1):
        for y in range(yc - n, yc + n + 1):
            ptss.append([x, y])
    return ptss

def cut_line_by_xmin(line, xmin):
    if line[0] > xmin and line[2] > xmin:
        return  True, line
    if line[0] <= xmin and line[2] <= xmin:
        return  False, line
    if abs(line[0] - line[2]) < 1:
        return  False, line
    # y = k*x  + b
    k = (line[3] - line[1]) / (line[2] - line[0])
    b = line[3] - k * line[2]
    y = k * xmin + b
    p0 = [xmin, y]
    if line[0] < line[2]:
        p1 = [line[2], line[3]]
    else:
        p1 = [line[0], line[1]]
    line = [p0[0], p0[1], p1[0], p1[1]]

    return True, line

def cut_line_by_xmax(line, xmax):
    if line[0] < xmax and line[2] < xmax:
        return  True, line
    if line[0] >= xmax and line[2] >= xmax:
        return  False, line
    if abs(line[0] - line[2]) < 1:
        return  False, line
    # y = k*x  + b
    k = (line[3] - line[1]) / (line[2] - line[0])
    b = line[3] - k * line[2]
    y = k * xmax + b
    p1 = [xmax, y]
    if line[0] > line[2]:
        p0 = [line[2], line[3]]
    else:
        p0 = [line[0], line[1]]
    return True, [p0[0], p0[1], p1[0], p1[1]]

def work_around_line(x0, y0, x1, y1, n=2, r=0.0, thickness=3):
    t = (thickness - 1) // 2
    # print("p:", x0, y0, x1, y1)
    if abs(x0 - x1) > abs(y0 - y1):
        ## y = k* x + b
        k = (y1 - y0) / (x1 - x0)
        b = y1 - k * x1

        ptss = []
        xc = (x0 + x1) / 2
        if n is None:
            n = int(abs(x1 - x0) * r)

        xmin = int(xc - n)
        xmax = int(xc + n)
        for x in range(xmin, xmax + 1):
            y = k * x + b
            for iy in range(thickness):
                ptss.append([x, y + t - iy])

        return ptss
    else:
        ## x = k* y + b
        k = (x1 - x0) / (y1 - y0)
        b = x1 - k * y1
        ptss = []

        yc = (y0 + y1) / 2
        if n is None:
            n = int(abs(y1 - y0) * r)
        ymin = int(yc - n)
        ymax = int(yc + n)

        for y in range(ymin, ymax + 1):
            x = k * y + b
            for ix in range(thickness):
                ptss.append([x + t - ix, y])
        return ptss


# def gen_TP_mask(norm_lines, h=256, w=256, with_ext=True):
#     """
#     1 cengter + 4  dis + 2
#     return [7, h, w]
#     """
#
#     # h, w, _ = img.shape
#
#     len_divide_v = np.sqrt(h ** 2 + w ** 2)
#
#     centermap = np.zeros((1, h, w), dtype=np.uint8)
#
#     displacement_map = np.zeros((4, h, w), dtype=np.float32)
#     length_map = np.zeros((1, h, w), dtype=np.float32)
#     degree_map = np.zeros((1, h, w), dtype=np.float32)
#
#     for l in norm_lines:
#         x0, y0, x1, y1 = w * l[0], h * l[1], w * l[2], h * l[3]
#
#         # print("p:", x0, y0, x1, y1)
#
#         xc = (x0 + x1) / 2
#         yc = (y0 + y1) / 2
#
#         if with_ext:
#             len_max = max(abs(x1 - x0), abs(y1 - y0))
#             # len_max = int(0.5 * len_max)
#             exp_pix = min(7, len_max)
#             ptss = work_around_line(x0, y0, x1, y1, n=exp_pix, thickness=1)
#
#             for p in ptss:
#                 xx = int(round(p[0]))
#                 yy = int(round(p[1]))
#
#                 xx = np.clip(xx, 0, w - 1)
#                 yy = np.clip(yy, 0, h - 1)
#
#                 sx = (1 - abs(xx - xc) / (2 * exp_pix))
#                 sy = (1 - abs(yy - yc) / (2 * exp_pix))
#
#                 centermap[0, yy, xx] = 255 * sx * sy
#
#                 x0d = x0 - xx
#                 y0d = y0 - yy
#                 x1d = x1 - xx
#                 y1d = y1 - yy
#
#                 displacement_map[0, yy, xx] = x0d
#                 displacement_map[1, yy, xx] = y0d
#                 displacement_map[2, yy, xx] = x1d
#                 displacement_map[3, yy, xx] = y1d
#
#         line_len, ang = line_len_and_angle(x0, y0, x1, y1)
#         line_len /= len_divide_v
#
#         x0d = x0 - xc
#         y0d = y0 - yc
#         x1d = x1 - xc
#         y1d = y1 - yc
#
#         # ptss = [
#         #     [int(np.floor(xc)), int(np.floor(yc))],
#         #     [int(np.ceil(xc)), int(np.ceil(yc))],
#         #     [int(np.floor(xc)), int(np.ceil(yc))],
#         #     [int(np.ceil(xc)), int(np.floor(yc))],
#         # ]
#         xc = int(round(xc))
#         yc = int(round(yc))
#         ptss = near_area_n(xc, yc, 3)
#
#         for p in ptss:
#             xx = int(round(p[0]))
#             yy = int(round(p[1]))
#
#             xx = np.clip(xx, 0, w - 1)
#             yy = np.clip(yy, 0, h - 1)
#
#             centermap[0, yy, xx] = 255
#             length_map[0, yy, xx] = line_len
#             degree_map[0, yy, xx] = ang
#
#             displacement_map[0, yy, xx] = x0d
#             displacement_map[1, yy, xx] = y0d
#             displacement_map[2, yy, xx] = x1d
#             displacement_map[3, yy, xx] = y1d
#
#     centermap[0, :, :] = cv2.GaussianBlur(centermap[0, :, :], (3, 3), 0.0)
#     centermap = np.array(centermap, dtype=np.float32) / 255.0
#     b = centermap.max() - centermap.min()
#     if b != 0:
#         centermap = (centermap - centermap.min()) / b
#
#     tp_mask = np.concatenate((centermap, displacement_map, length_map, degree_map), axis=0)
#     return tp_mask

def gen_TP_mask2(norm_lines,  h = 256, w = 256, with_ext=False):
    """
    1 cengter + 4  dis + 2
    return [7, h, w]
    """

    #h, w, _ = img.shape

    len_divide_v = np.sqrt(h**2 + w**2)
    radius = 1

    centermap = np.zeros((1, h, w), dtype=np.uint8)
    #displacement_map = -np.ones((4, h, w), dtype=np.float32) * 1000.0

    displacement_map = np.zeros((4, h, w), dtype=np.float32)
    length_map = np.zeros((1, h, w), dtype=np.float32)
    degree_map = np.zeros((1, h, w), dtype=np.float32)

    for l in norm_lines:
        x0 = int(round(l[0] * w))
        y0 = int(round(l[1] * h))
        x1 = int(round(l[2] * w))
        y1 = int(round(l[3] * h))

        xc = round(w * (l[0] + l[2]) / 2)
        yc = round(h * (l[1] + l[3]) / 2)

        xc = int(np.clip(xc, 0, w - 1))
        yc = int(np.clip(yc, 0, h - 1))

        centermap[0, yc, xc] = 255

        line_len, ang = line_len_and_angle(x0, y0, x1, y1)
        line_len /= len_divide_v
        length_map[0, yc, xc] = line_len
        degree_map[0, yc, xc] = ang

        x0d = x0 - xc
        y0d = y0 - yc
        x1d = x1 - xc
        y1d = y1 - yc

        #print('x0d: ', x0d)

        displacement_map[0, yc, xc] = x0d  # / 2
        displacement_map[1, yc, xc] = y0d  # / 2
        displacement_map[2, yc, xc] = x1d  # / 2
        displacement_map[3, yc, xc] = y1d  # / 2

        ## walk around line
        #ptss = work_around_line(x0, y0, x1, y1, n=5, r=0.0, thickness=3)

        # extrapolated to a 3×3 window
        ptss = near_area_n(xc, yc, n=3)
        for p in ptss:
            xc = round(p[0])
            yc = round(p[1])
            xc = int(np.clip(xc, 0, w - 1))
            yc = int(np.clip(yc, 0, h - 1))
            # x0d = x0 - xc
            # y0d = y0 - yc
            # x1d = x1 - xc
            # y1d = y1 - yc
            displacement_map[0, yc, xc] = x0d# / 2
            displacement_map[1, yc, xc] = y0d# / 2
            displacement_map[2, yc, xc] = x1d# / 2
            displacement_map[3, yc, xc] = y1d# / 2

            length_map[0, yc, xc] = line_len
            degree_map[0, yc, xc] = ang

        xc = round(w * (l[0] + l[2]) / 2)
        yc = round(h * (l[1] + l[3]) / 2)

        xc = int(np.clip(xc, 0, w - 1))
        yc = int(np.clip(yc, 0, h - 1))

        centermap[0, yc, xc] = 255

        line_len, ang = line_len_and_angle(x0, y0, x1, y1)
        line_len /= len_divide_v
        length_map[0, yc, xc] = line_len
        degree_map[0, yc, xc] = ang

        x0d = x0 - xc
        y0d = y0 - yc
        x1d = x1 - xc
        y1d = y1 - yc

        displacement_map[0, yc, xc] = x0d  # / 2
        displacement_map[1, yc, xc] = y0d  # / 2
        displacement_map[2, yc, xc] = x1d  # / 2
        displacement_map[3, yc, xc] = y1d  # / 2

    centermap[0, :, :] = cv2.GaussianBlur(centermap[0, :, :], (3,3), 0.0)
    centermap = np.array(centermap, dtype=np.float32) / 255.0
    b = centermap.max() - centermap.min()
    if b !=0:
        centermap = ( centermap - centermap.min() ) / b

    tp_mask = np.concatenate((centermap, displacement_map, length_map, degree_map), axis=0)
    return tp_mask


def get_ext_lines(norm_lines, h=256, w=256, min_len=0.125):
    mu_half = min_len / 2
    ext_lines = []
    for line in norm_lines:
        x0, y0, x1, y1 = line
        line_len = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        nn = int(line_len / mu_half) - 1
        # print("nn: ", nn)
        if nn <= 1:
            ext_lines.append(line)
        else:
            ## y = k * x + b
            if abs(x0 - x1) > abs(y0 - y1):
                ## y = k* x + b
                k = (y1 - y0) / (x1 - x0)
                b = y1 - k * x1
                step = (x1 - x0) / (nn + 1)
                len_step = 2 * step  # (x1 - x0) / (nn - 1)
                for ix in range(nn):
                    ix0 = x0 + ix * step
                    # ix1 = x0 + (ix + 1) * step
                    ix1 = ix0 + len_step
                    iy0 = k * ix0 + b
                    iy1 = k * ix1 + b
                    ext_lines.append([ix0, iy0, ix1, iy1])

            else:
                ## x = k* y + b
                k = (x1 - x0) / (y1 - y0)
                b = x1 - k * y1
                step = (y1 - y0) / (nn + 1)
                len_step = 2 * step  # (y1 - y0) / (nn - 1)
                for iy in range(nn):
                    iy0 = y0 + iy * step
                    # iy1 = y0 + (iy + 1) * step
                    iy1 = iy0 + len_step
                    ix0 = k * iy0 + b
                    ix1 = k * iy1 + b
                    ext_lines.append([ix0, iy0, ix1, iy1])
    # print("ext_lines: ", len(ext_lines))
    return ext_lines

def gen_SOL_map(norm_lines,  h =256, w =256, min_len =0.125, with_ext= False):
    """
    1 + 4 + 2
    return [7, h, w]
    """
    ext_lines = get_ext_lines(norm_lines, h, w, min_len)
    return gen_TP_mask2(ext_lines, h, w, with_ext), ext_lines


def gen_junction_and_line_mask(norm_lines, h = 256, w = 256):
    junction_map = np.zeros((h, w, 1), dtype=np.float32)
    line_map = np.zeros((h, w, 1), dtype=np.float32)

    radius = 1
    for l in norm_lines:
        x0 = int(round(l[0] * w))
        y0 = int(round(l[1] * h))
        x1 = int(round(l[2] * w))
        y1 = int(round(l[3] * h))
        cv2.line(line_map, (x0, y0), (x1, y1), (255, 255, 255), radius)
        #cv2.circle(junction_map, (x0, y0), radius, (255, 255, 255), radius)
        #cv2.circle(junction_map, (x1, y1), radius, (255, 255, 255), radius)
        
        ptss = near_area_n(x0, y0, n=3)
        ptss.extend( near_area_n(x1, y1, n=3) )
        for p in ptss:
            xc = round(p[0])
            yc = round(p[1])
            xc = int(np.clip(xc, 0, w - 1))
            yc = int(np.clip(yc, 0, h - 1))
            junction_map[yc, xc, 0] = 255

    junction_map[:, :, 0] = cv2.GaussianBlur(junction_map[:, :, 0], (3,3), 0.0)
    junction_map = np.array(junction_map, dtype=np.float32) / 255.0
    b = junction_map.max() - junction_map.min()
    if b !=0:
        junction_map = ( junction_map - junction_map.min() ) / b
    # line map use binary one
    line_map = np.array(line_map, dtype=np.float32) / 255.0
#     line_map[:, :, 0] = cv2.GaussianBlur(line_map[:, :, 0], (3, 3), 0.0)
#     line_map = np.array(line_map, dtype=np.float32) / 255.0
#     b = line_map.max() - line_map.min()
#     if b !=0:
#         line_map = ( line_map - line_map.min() ) / b

    return junction_map, line_map
