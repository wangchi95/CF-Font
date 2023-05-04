import torch


def naive_dt(mask_fg, max_n = -1): # fg 0 bg 1-n
    b,c,h,w = mask_fg.shape
    assert c == 1
    mask_distance = mask_fg * 1.0
    mask_n = h * w
    bg_n = 1
    i = 1
    kernel = torch.tensor([[0.,1.,0.], [1.,1.,1.], [0.,1.,0.]], device=mask_fg.device)[None, None]
    while bg_n > 0 and i != max_n:
        mask_fg = torch.nn.functional.conv2d(mask_fg.float(), kernel, padding=1) > 0
        mask_distance += 1.0 * mask_fg
        bg_n = torch.logical_not(mask_fg).sum()
        i += 1
    return i - mask_distance

def PHL(f1, f2, thres=0.5): # N,1,H,W fg1 bg0
    # f1, f2 in [-1, 1] fg -1 bg 1
    # Pseudo_Hamming_Loss
    mask1_fg = f1 < thres # black
    mask2_fg = f2 < thres # TODO Gradient???
    prob1_fg = (1-f1) / 2
    prob2_fg = (1-f2) / 2
    dis1 = naive_dt(mask1_fg) * prob1_fg
    dis2 = naive_dt(mask2_fg) * prob2_fg
    mask1minus2 = 1.0 * mask1_fg - 1.0 * mask2_fg
    mask1not2 = mask1minus2 > 0
    mask2not1 = mask1minus2 < 0
    dis = torch.zeros_like(dis1)
    dis[mask2not1] = dis1[mask2not1]
    dis[mask1not2] = dis2[mask1not2]
    return dis