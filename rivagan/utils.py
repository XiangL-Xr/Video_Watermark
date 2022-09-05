import os
import cv2
import zlib
import torch
import subprocess
import torchvision.utils

from math import exp
from torch.nn.functional import conv2d
from tempfile import NamedTemporaryFile
from glob import glob
from PIL import Image
from torchvision import transforms

def gaussian(window_size, sigma):
    """Gaussian window.

    https://en.wikipedia.org/wiki/Window_function#Gaussian_window
    """
    _exp = [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    gauss = torch.Tensor(_exp)
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):

    padding_size = window_size // 2

    mu1 = conv2d(img1, window, padding=padding_size, groups=channel)
    mu2 = conv2d(img2, window, padding=padding_size, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, padding=padding_size, groups=channel) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=padding_size, groups=channel) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, padding=padding_size, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    _ssim_quotient = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    _ssim_divident = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = _ssim_quotient / _ssim_divident

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def psnr(img1, img2):
    mse = torch.mean( (img1 - img2) ** 2 )
    return 20 * torch.log10(2.0 / torch.sqrt(mse))

def mjpeg(x):
    """
    Write each video to disk and re-read it from disk.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """
    y = torch.zeros(x.size())
    _, _, _, height, width = x.size()

    for n in range(x.size(0)):
        tempfile = NamedTemporaryFile(suffix=".avi")

        vout = cv2.VideoWriter(tempfile.name, cv2.VideoWriter_fourcc(*'H264'), 20.0, (width, height))
        for l in range(x.size(2)):
            image = x[n,:,l,:,:] # (3, H, W)
            image = torch.clamp(image.permute(1,2,0), min=-1.0, max=1.0)
            vout.write(((image + 1.0) * 127.5).detach().cpu().numpy().astype("uint8"))
        vout.release()

        vin = cv2.VideoCapture(tempfile.name)
        for l in range(x.size(2)):
            _, frame = vin.read() # (H, W, 3)
            frame = frame / 127.5 - 1.0
            frame = torch.tensor(frame)
            y[n,:,l,:,:] = frame.permute(2,0,1)
        
        tempfile.close()
    return y.to(x.device)

def encoder_h264(x, tmp='.tmp'):
    y = torch.zeros(x.size())
    _, _, _, height, width = x.size()

    #print('---x_size', x.size())
    for n in range(x.size(0)):
        if os.path.exists(tmp):
            FNULL = open(os.devnull, 'w')
            subprocess.call(f"rm {tmp}/*.png", shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
        else:
            subprocess.call(f"mkdir {tmp}", shell=True)

        tempfile = NamedTemporaryFile(suffix=".avi")
        for l in range(x.size(2)):
            image = x[n, :, l, :, :]
            image = torch.clamp(image, min = -1.0, max = 1.0)
            image = (image + 1.0) * 127.5
            filename = os.path.join(tmp, '{:03d}.png'.format(l+1))
            torchvision.utils.save_image(image, filename, x.shape[2], normalize = False)
        
        # 将经过水印处理后的图片拼接成视频
        subprocess.call(f"ffmpeg -y -r 20.0 -i {tmp}/%03d.png -vcodec h264 -preset ultrafast -qscale 0 -pix_fmt yuv420p -v quiet {tempfile.name}", shell=True)
        
        if len(os.listdir(tmp)) > 0:
            FNULL = open(os.devnull, 'w')
            subprocess.call(f"rm {tmp}/*.png", shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
        
        # 将视频抽帧为图片
        subprocess.call(f"ffmpeg -i {tempfile.name} -q 1 -v quiet {tmp}/%03d.png", shell=True)

        for idx, img_path in enumerate(glob(f"{tmp}/*.png")):
            
            frame = Image.open(img_path).convert('RGB')
            frame = transforms.ToTensor()(frame)
            frame = frame / 127.5 - 1.0
            #print("frame_shape:", frame.shape)
            y[n,:,l,:,:] = frame
    
    return y.to(x.device)
