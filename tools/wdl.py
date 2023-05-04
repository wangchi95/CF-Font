import torch
import kornia as K

def w_dis_1d(a, b): # [N, C]
    a_cdf = torch.cumsum(a, dim=1) # [N, C]
    a_cdf = a_cdf / a_cdf[:,-1:]
    b_cdf = torch.cumsum(b, dim=1) 
    b_cdf = b_cdf / b_cdf[:,-1:]
    return (a_cdf - b_cdf).abs().sum(1) # [N]

def WDL(f1, f2): # N,1,H,W
    # f1, f2 in [-1, 1] fg -1 bg 1
    # W distance Loss
    # HW
    f1 = (1-f1) / 2 # fg 1 bg 0
    f2 = (1-f2) / 2 # fg 1 bg 0
    B = f1.shape[0]
    f1_0 = f1.sum((1,2)) # N,W
    f2_0 = f2.sum((1,2)) # N,W
    loss_0 = w_dis_1d(f1_0, f2_0) # N
    f1_1 = f1.sum((1,3)) # N,H
    f2_1 = f2.sum((1,3)) # N,H
    loss_1 = w_dis_1d(f1_1, f2_1) # N
    losses = [loss_0, loss_1]
    for angle in [15., 30., 45., 60., 75.]:
        f1r = K.geometry.rotate(f1, angle * torch.ones(B, device=f1.device))
        f2r = K.geometry.rotate(f2, angle * torch.ones(B, device=f1.device))
        f1r_0 = f1r.sum((1,2)) # N,W
        f2r_0 = f2r.sum((1,2)) # N,W
        lossr_0 = w_dis_1d(f1r_0, f2r_0) # N
        losses.append(lossr_0)
        f1r_1 = f1r.sum((1,3)) # N,H
        f2r_1 = f2r.sum((1,3)) # N,H
        lossr_1 = w_dis_1d(f1r_1, f2r_1) # N
        losses.append(lossr_1)
    loss = torch.stack(losses).mean()
    
    return loss
