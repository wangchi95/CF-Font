import numpy as np
import lpips

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error as mse
    from skimage.metrics import peak_signal_noise_ratio as psnr
except:
    from skimage.measure import compare_ssim as ssim
    from skimage.measure import compare_mse as mse
    from skimage.measure import compare_psnr as psnr
    
class LPIPS():
    def __init__(self, using_gpu=False):
        self.model_lpips = lpips.LPIPS(net='alex')
        self.using_gpu = using_gpu
        if using_gpu:
            self.model_lpips.cuda()
    
    def cal_lpips(self, i0, i1):
        img0 = lpips.im2tensor(i0) # [-1, 1]
        img1 = lpips.im2tensor(i1)
        if self.using_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()

        # Compute distance
        dist01 = self.model_lpips.forward(img0,img1).flatten() # RGB image from [-1,1]
        assert len(dist01) == 1
        return dist01[0]

L1 = lambda in0, in1, data_range=255.: np.mean(np.abs((in0 / data_range - in1 / data_range))) # HWC, [0,1] # Smaller, better
RMSE = lambda in0, in1, data_range=255.: mse(in0 / data_range, in1 / data_range) ** 0.5  # Smaller, better

# SSIM = lambda in0, in1, data_range=255.: ssim(in0, in1, data_range=data_range, multichannel=True)  # Bigger, better
#SSIM = lambda in0, in1, data_range=255.: ssim(in0, in1, data_range=data_range, channel_axis=True)  # Bigger, better

def SSIM(imgs_fake, imgs_real):
    mssim0 = ssim(imgs_fake[:,:,0], imgs_real[:,:,0], data_range=255, gaussian_weights=True)
    mssim1 = ssim(imgs_fake[:,:,1], imgs_real[:,:,1], data_range=255, gaussian_weights=True)
    mssim2 = ssim(imgs_fake[:,:,2], imgs_real[:,:,2], data_range=255, gaussian_weights=True)
    mssim = (mssim0 + mssim1 + mssim2)/3
    return mssim

PSNR = lambda gt, pred, data_range=255.: psnr(gt, pred, data_range=data_range)  # Bigger, better

# FID: python -m pytorch_fid path/to/dataset1 path/to/dataset2
