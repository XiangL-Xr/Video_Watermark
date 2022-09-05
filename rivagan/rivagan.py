# !/usr/bin/python3
# coding: utf-8

import os, sys
import gc, cv2
import math
import json, copy
import random, time
import shlex
import torch
import subprocess

import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils

from tqdm import tqdm
from glob import glob
from itertools import chain
from torchvision import transforms

from rivagan.adversary import Adversary, Critic
from rivagan.attention import AttentiveEncoder, AttentiveDecoder
from rivagan.dense import DenseEncoder, DenseDecoder
from rivagan.dataloader import load_train_val
from rivagan.noise import Crop, Scale, Compression
from rivagan.utils import mjpeg, ssim, psnr, encoder_h264

def get_acc(y_true, y_pred):
    assert y_true.size() == y_pred.size()
    return (y_pred >= 0.0).eq(y_true >= 0.5).sum().float().item() / y_pred.numel()

def quantize(frames):
    # [-1.0, 1.0] -> {0, 255} -> [-1.0, 1.0]
    return ((frames + 1.0) * 127.5).int().float() / 127.5 - 1.0

def make_pair(frames, data_dim, use_bit_inverse=True, multiplicity=1):
    # Add multiplicity to further stabilize training.
    frames = torch.cat([frames] * multiplicity, dim=0).cuda()
    data = torch.zeros((frames.size(0), data_dim)).random_(0, 2).cuda()
    # Add the bit-inverse to stabilize training.
    if use_bit_inverse:
        frames = torch.cat([frames, frames], dim=0).cuda()
        data = torch.cat([data, 1.0 - data], dim=0).cuda()

    return frames, data

class RivaGAN(object):
    def __init__(self, model="attention", data_dim=32):
        self.model = model
        self.data_dim = data_dim
        self.adversary = Adversary().cuda()
        self.critic = Critic().cuda()
        if model == "attention":
            self.encoder = AttentiveEncoder(data_dim=data_dim).cuda()
            self.decoder = AttentiveDecoder(self.encoder).cuda()
            #print('--encoder model:', self.encoder)
            #print('--decoder model:', self.decoder)
        elif model == "dense":
            self.encoder = DenseEncoder(data_dim=data_dim).cuda()
            self.decoder = DenseDecoder(data_dim=data_dim).cuda()
        else:
            raise ValueError("Unknown model: %s" % model)

    def fit(self, dataset, log_dir=False, 
                seq_len=1, batch_size=12, lr=5e-4, 
                use_critic=False, use_adversary=False, 
                epochs=300, use_bit_inverse=True, use_noise=True):
        if not log_dir:
            log_dir = "experiments/%s-%s" % (self.model, str(int(time.time())))
        os.makedirs(log_dir, exist_ok=False)

        # Set up the noise layers
        crop = Crop()
        scale = Scale()
        compress = Compression()
        def noise(frames):
            if use_noise:
                if random.random() < 0.5:
                    frames = crop(frames)
                if random.random() < 0.5:
                    frames = scale(frames)
                if random.random() < 0.5:
                    frames = compress(frames)
            return frames

        # Set up the data and optimizers
        train, val = load_train_val(seq_len, batch_size, dataset)
        G_opt = optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=lr)
        G_scheduler = optim.lr_scheduler.ReduceLROnPlateau(G_opt)
        D_opt = optim.Adam(chain(self.adversary.parameters(), self.critic.parameters()), lr=lr)
        D_scheduler = optim.lr_scheduler.ReduceLROnPlateau(D_opt)

        # Set up the log directory
        with open(os.path.join(log_dir, "config.json"), "wt") as fout:
            fout.write(json.dumps({
                        "model"     : self.model, 
                        "data_dim"  : self.data_dim,
                        "seq_len"   : seq_len,
                        "batch_size": batch_size,
                        "dataset"   : dataset,
                        "lr"        : lr,
                        "log_dir"   : log_dir,
                      }, indent=2, default=lambda o: str(o)))

        # Optimize the model
        history = []
        for epoch in range(1, epochs + 1):
            metrics = {"train.loss"     : [], 
                       "train.raw_acc"  : [], 
                       "train.mjpeg_acc": [], 
                       "train.adv_loss" : [],
                       "val.ssim"       : [],
                       "val.psnr"       : [],
                       "val.crop_acc"   : [], 
                       "val.scale_acc"  : [], 
                       "val.mjpeg_acc"  : [], 
                      }
            gc.collect()
            self.encoder.train()
            self.decoder.train()

            # Optimize critic-adversary
            if use_critic or use_adversary:
                iterator = tqdm(train, ncols=0)
                for frames in iterator:
                    frames, data = make_pair(frames, self.data_dim, use_bit_inverse=use_bit_inverse)
                    wm_frames = self.encoder(frames, data)
                    adv_loss = 0.0
                    if use_critic:
                        adv_loss += torch.mean(self.critic(frames) - self.critic(wm_frames))
                    if use_adversary:
                        adv_loss -= F.binary_cross_entropy_with_logits(self.decoder(self.adversary(wm_frames)), data)
                    D_opt.zero_grad()
                    adv_loss.backward()
                    D_opt.step()
                    for p in self.critic.parameters():
                        p.data.clamp_(-0.1, 0.1)
                    metrics["train.adv_loss"].append(adv_loss.item())
                    iterator.set_description("Adversary | %s" % np.mean(metrics["train.adv_loss"]))

            # Optimize encoder-decoder using critic-adversary
            if use_critic or use_adversary:
                iterator = tqdm(train, ncols=0)
                for frames in iterator:
                    frames, data = make_pair(frames, self.data_dim, use_bit_inverse=use_bit_inverse)
                    wm_frames = self.encoder(frames, data)
                    loss = 0.0
                    if use_critic:
                        critic_loss = torch.mean(self.critic(wm_frames))
                        loss += 0.1 * critic_loss
                    if use_adversary:
                        adversary_loss = F.binary_cross_entropy_with_logits(self.decoder(self.adversary(wm_frames)), data)
                        loss += 0.1 * adversary_loss
                    G_opt.zero_grad()
                    loss.backward()
                    G_opt.step()

            # Optimize encoder-decoder
            iterator = tqdm(train, ncols=0)
            for frames in iterator:
                frames, data = make_pair(frames, self.data_dim, use_bit_inverse=use_bit_inverse)

                wm_frames = self.encoder(frames, data)
                wm_raw_data = self.decoder(noise(wm_frames))
                wm_mjpeg_data = self.decoder(mjpeg(wm_frames))
                #wm_mjpeg_data = self.decoder(encoder_h264(wm_frames))

                loss = 0.0
                loss += F.binary_cross_entropy_with_logits(wm_raw_data, data)
                loss += F.binary_cross_entropy_with_logits(wm_mjpeg_data, data)
                G_opt.zero_grad()
                loss.backward()
                G_opt.step()

                metrics["train.loss"].append(loss.item())
                metrics["train.raw_acc"].append(get_acc(data, wm_raw_data))
                metrics["train.mjpeg_acc"].append(get_acc(data, wm_mjpeg_data))
                iterator.set_description("%s | Loss %.3f | Raw %.3f | MJPEG %.3f" % (
                    epoch, 
                    np.mean(metrics["train.loss"]), 
                    np.mean(metrics["train.raw_acc"]),
                    np.mean(metrics["train.mjpeg_acc"]),
                ))

            # Validate
            gc.collect()
            self.encoder.eval()
            self.decoder.eval()
            iterator = tqdm(val, ncols=0)
            with torch.no_grad():
                for frames in iterator:
                    frames = frames.cuda()
                    data = torch.zeros((frames.size(0), self.data_dim)).random_(0, 2).cuda()

                    wm_frames = self.encoder(frames, data)
                    wm_crop_data = self.decoder(mjpeg(crop(wm_frames)))
                    wm_scale_data = self.decoder(mjpeg(scale(wm_frames)))
                    wm_mjpeg_data = self.decoder(mjpeg(wm_frames))

                    metrics["val.ssim"].append(ssim(frames[:,:,0,:,:], wm_frames[:,:,0,:,:]).item())
                    metrics["val.psnr"].append(psnr(frames[:,:,0,:,:], wm_frames[:,:,0,:,:]).item())
                    metrics["val.crop_acc"].append(get_acc(data, wm_crop_data))
                    metrics["val.scale_acc"].append(get_acc(data, wm_scale_data))
                    metrics["val.mjpeg_acc"].append(get_acc(data, wm_mjpeg_data))

                    iterator.set_description( \
                                "%s | SSIM %.3f | PSNR %.3f | Crop %.3f | Scale %.3f | MJPEG %.3f" % (
                                                epoch, 
                                                np.mean(metrics["val.ssim"]),
                                                np.mean(metrics["val.psnr"]),
                                                np.mean(metrics["val.crop_acc"]),
                                                np.mean(metrics["val.scale_acc"]),
                                                np.mean(metrics["val.mjpeg_acc"]),
                    ))

            metrics = {k: round(np.mean(v), 3) if len(v) > 0 else "NaN" for k, v in metrics.items()}
            metrics["epoch"] = epoch
            history.append(metrics)
            pd.DataFrame(history).to_csv(os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
            with open(os.path.join(log_dir, "metrics.json"), "wt") as fout:
                fout.write(json.dumps(metrics, indent=2, default=lambda o: str(o)))
            
            torch.save(self, os.path.join(log_dir, "model.pt"))
            G_scheduler.step(metrics["train.loss"])
        
        return history
    
    def save(self, path_to_model):
        torch.save(self, path_to_model)

    def load(path_to_model):
        return torch.load(path_to_model)
    
    def get_mark_flag(self, water_marking, decoded_messages):
        ### If input video is a randomly cropped segment """
        ### First, find the initial frame with embedded watermark """
        ### Then, extracting the watermark of every embedded image frame
        
        #print('---decoded_messages:', decoded_messages)
        expand_value = decoded_messages.repeat(water_marking.shape[0], axis = 0)
        bitwise_avg_err = np.sum(np.abs(expand_value - water_marking), axis=1) / water_marking.shape[1]
        bitwise_err = np.min(bitwise_avg_err)
        
        mark_flag = False
        if bitwise_err < 0.15:
            mark_flag = True
            
        return mark_flag
    
    def encode(self, video_in, data, video_out, video_idx, fps=-1, tmp='.en_tmp'):
        assert data.shape[1] == self.data_dim
        has_audio = False
        #print("=" * 80)
        print(f"=> 处理视频路径： {video_in}")
        args = shlex.split(f"ffprobe -v quiet -print_format json -show_streams {video_in}")
        video_meta = subprocess.check_output(args).decode('utf-8')
        video_meta = json.loads(video_meta)
        
        if video_meta['streams'][0]['codec_type'] == "audio":
            print(f"=> audio in stream 0, exchange...")
            tmp_data = video_meta['streams'][0]
            video_meta['streams'].pop(0)
            video_meta['streams'].append(tmp_data)
        
        fps_split = video_meta['streams'][0]['r_frame_rate'].split('/')
        video_fps = round(int(fps_split[0]) / int(fps_split[1]), 2)
        #video_fps = int(video_meta['streams'][0]['r_frame_rate'].split('/')[0])
        bit_rate = int(video_meta['streams'][0]['bit_rate'])
        
        ### audio parames
        if len(video_meta['streams']) > 1:
            #print(f"=> video has audio ...")
            has_audio = True
            #print('----', video_meta['streams'][1])
            if 'sample_rate' in video_meta['streams'][1]:
                audio_sample_rate = int(video_meta['streams'][1]['sample_rate'])
                audio_channels    = int(video_meta['streams'][1]['channels'])
                audio_bit_rate    = int(video_meta['streams'][1]['bit_rate'])
            elif 'sample_rate' in video_meta['streams'][2]:
                audio_sample_rate = int(video_meta['streams'][2]['sample_rate'])
                audio_channels    = int(video_meta['streams'][2]['channels'])
                audio_bit_rate    = int(video_meta['streams'][2]['bit_rate'])

        print(f"=> 图片分辨率: {video_meta['streams'][0]['height']}, {video_meta['streams'][0]['width']}")
        print(f"=> 视频fps: {video_fps}")
        print(f"=> 视频时长: {video_meta['streams'][0]['duration']} 秒")
        print(f"=> 编码方式: {video_meta['streams'][0]['codec_name']}")
        print("-" * 60)

        if os.path.exists(tmp):
            FNULL = open(os.devnull, 'w')
            subprocess.call(f"rm {tmp}/*.png", shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
        else:
            subprocess.call(f"mkdir {tmp}", shell=True)        
        if fps == -1 or fps > video_fps:
            fps = video_fps
        
        ### extracting audios
        if has_audio:
            print(f"=> 从视频中提取音频信息...")
            audio_path = os.path.join(tmp, 'tmp_audio.wav')
            if os.path.exists(audio_path):
                os.remove(audio_path)
            subprocess.call(f"ffmpeg -i {video_in} -vn -ar {audio_sample_rate} -ac {audio_channels} -ab {audio_bit_rate} -f wav -v quiet {audio_path}", shell=True)
        
        ### extracting images
        print(f"=> 从视频中提取图像帧...")
        subprocess.call(f"ffmpeg -i {video_in} -q 1 -v quiet {tmp}/%04d.png", shell=True)
        
        ### watermark embedded
        start_time = time.time()
        mark_frame_n = 0
        for frame_idx, img_path in enumerate(glob(f"{tmp}/*.png")):
            #if frame_idx > 10 * fps:
            #    continue
            if frame_idx % 5 == 0:
                img_path_copy = img_path[:-4]+"_bak.png"
                subprocess.call(f"cp {img_path} {img_path_copy}", shell=True)
            
                frame_read = time.time()
                frame = cv2.imread(img_path_copy)    
                frame = torch.FloatTensor([frame]) / 127.5 - 1.0
                frame = frame.permute(3, 0, 1, 2).unsqueeze(0).cuda()
                wm_frame = self.encoder(frame, data)
                wm_frame = torch.clamp(wm_frame, min=-1.0, max=1.0)
                wm_frame = ((wm_frame[0,:,0,:,:].permute(1,2,0) + 1.0) * 127.5)
                wm_frame = wm_frame.detach().cpu().numpy().astype("uint8")
                cv2.imwrite(img_path, wm_frame)
                
                mark_frame_n += 1
                print("=> 第 [{}/{}] 视频帧加水印完成 ...".format(video_idx, frame_idx))

        print("-" * 60)
        print(f"=> 第 {video_idx} 个视频加水印的总帧数：{mark_frame_n}")
        
        video_start = time.time()
        # 将经过水印处理后的图片拼接成视频
        target_bit_rate = bit_rate * 1.02
        if has_audio:
            subprocess.call(f"ffmpeg -y -i {audio_path} -r {video_fps} -i {tmp}/%04d.png -vcodec h264 -profile main -b:v {target_bit_rate} -pix_fmt yuv420p -v quiet {video_out}", shell=True)
        else:
            subprocess.call(f"ffmpeg -y -r {video_fps} -i {tmp}/%04d.png -vcodec h264 -profile main -b:v {target_bit_rate} -pix_fmt yuv420p -v quiet {video_out}", shell=True)
        
        print(f"=> 加水印的总时长：{time.time() - start_time} s")
        print(f"=> 加水印后视频保存至: {video_out}")
        print("=" * 80)
   
    def extract_from_img(self, img_path):
        frame = cv2.imread(img_path)
        frame = torch.FloatTensor([frame]) / 127.5 - 1.0            # (L, H, W, 3)
        frame = frame.permute(3, 0, 1, 2).unsqueeze(0).cuda()       # (1, 3, L, H, W)
        decoded_messages = self.decoder(frame)[0].unsqueeze(0).detach().cpu().numpy()
        decoded_rounded = decoded_messages.round().clip(0, 1)

        return decoded_rounded
    
    def decode(self, video_in, video_idx, water_marking, fps = -1, tmp = '.de_tmp'):        
        # 获取视频的属性
        #print("=" * 80)
        print(f"=> 处理视频路径: {video_in}")
        args = shlex.split(f"ffprobe -v quiet -print_format json -show_streams {video_in}")
        video_meta = subprocess.check_output(args).decode('utf-8')
        video_meta = json.loads(video_meta)
        
        fps_split = video_meta['streams'][0]['r_frame_rate'].split('/')
        video_fps = round(int(fps_split[0]) / int(fps_split[1]), 2)
        #video_fps = int(video_meta['streams'][0]['r_frame_rate'].split('/')[0])

        print(f"=> 图片分辨率: {video_meta['streams'][0]['height']}, {video_meta['streams'][0]['width']}")
        print(f"=> 视频fps:    {video_fps}")
        print(f"=> 视频时长:   {video_meta['streams'][0]['duration']} 秒")
        print(f"=> 编码方式:   {video_meta['streams'][0]['codec_name']}")
        print("-" * 60)

        if os.path.exists(tmp):
            FNULL = open(os.devnull, 'w')
            subprocess.call(f"rm {tmp}/*.png", shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
        else:
            subprocess.call(f"mkdir {tmp}", shell=True)        
        if fps == -1 or fps > video_fps:
            fps = video_fps
        
        subprocess.call(f"ffmpeg -y -i {video_in} -r {fps} -q 1 -v quiet {tmp}/%04d.png", shell=True)
        
        start_time = time.time()
        extract_frame_n = 0
        decoded_watermark_list = []
        
        """
        mark_flag_dict = {}
        for frame_idx, img_path in enumerate(glob(f"{tmp}/*.png")):     
            if frame_idx <= 10:
                decoded_rounded = self.extract_from_img(img_path)
                mark_flag = self.get_mark_flag(water_marking, decoded_rounded)
                mark_flag_dict[frame_idx] = mark_flag      # length of mark_flag_dict: 11
            else:
                break

        init_frame_idx = -1
        #for key, value in mark_flag_dict.items():
        for key in range(len(mark_flag_dict)-5):
            if mark_flag_dict[key] == mark_flag_dict[key+5] == True:
                init_frame_idx = key
                break
            
        if init_frame_idx == -1:
            print("=> 未检测到含水印的初始帧，疑似未嵌入水印的原始视频 ...")
     
        flag = True if init_frame_idx % 5 == 0 else False
        """
        
        for frame_idx, img_path in enumerate(glob(f"{tmp}/*.png")):
            if frame_idx > 10 * fps:
                continue

            if frame_idx % 5 == 0:
                decoded_rounded = self.extract_from_img(img_path)
                extract_frame_n += 1                
                decoded_watermark_list.append(decoded_rounded)
                print("=> 第 [{}/{}] 视频帧提取水印完成 ...".format(video_idx, frame_idx))
            
            #elif (not flag) and frame_idx % 5 == init_frame_idx:
            #    decoded_rounded = self.extract_from_img(img_path)
            #    extract_frame_n += 1
            #    decoded_watermark_list.append(decoded_rounded)
            #    print("=> 第 [{}/{}] 视频帧提取水印完成 ...".format(video_idx, frame_idx))
        
        print("-" * 60)
        print(f"=> 第 {video_idx} 个视频提取水印的总帧数: {extract_frame_n}")
        print(f"=> 提取水印的总时长：{time.time() - start_time} s")
        print("=" * 80)

        return decoded_watermark_list
