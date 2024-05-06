from pytorch_msssim import ssim
from torchvision.io import read_image as read_image
import lpips
import torch
import os


dir_1 = "output_fft"
dir_2 = "output_dct_no_corner"

lpips_loss_1 =  0
lpips_loss_2 =  0

ssim_loss_1 = 0
ssim_loss_2 = 0
loss_fn_alex = lpips.LPIPS(net='alex')


for i in range(50):
    print(i)
    img_gt = read_image(os.path.join(dir_1, str(i), "no_w_image.png"))[None, :]
    img_1 = read_image(os.path.join(dir_1, str(i), "w_image.png"))[None,:]
    img_2 = read_image(os.path.join(dir_2, str(i), "w_image.png"))[None, :]
    
    ssim_loss_1 += 1 - ssim(img_gt.float() , img_1.float(), data_range=255, size_average=True)
    ssim_loss_2 += 1 - ssim(img_gt.float() , img_2.float(), data_range=255, size_average=True)
    
    lpips_loss_1 += loss_fn_alex(img_gt, img_1)
    lpips_loss_2 += loss_fn_alex(img_gt, img_2)


print(f"Average ssim loss 1 : {ssim_loss_1/50} :: Average ssim loss 2 :: {ssim_loss_2/50}")
print(f"Average lpips loss 1 : {lpips_loss_1/50} :: Average lpips loss 2 :: {lpips_loss_2/50}")








