# !/usr/bin/python3
# coding: utf-8

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import pickle
import random
import torch
import argparse
import numpy as np

from collections import Counter
from rivagan import RivaGAN

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='./data',
                    help='the path to the project root directory.')
parser.add_argument('--checkpoint', type=str, default='./checkpoints/model_h264_epoch300.pt',
                    help='the path of load pre-trained model weights')
parser.add_argument('--watermark_path', type=str, default='./data/water_marking/water_marking_32.pickle',
                    help='the path of 32-bits water marking')
parser.add_argument('--input_videos_dir', type=str, default='./data/input_videos',
                    help='the path of input videos')
parser.add_argument('--output_dir', type=str, default='./data/results_test',
                    help='the path of output results after add watermark messages')
parser.add_argument('--label_path', type=str, default=' ',
                    help='the path of videomark encode label')
parser.add_argument('--threshold', type=float, default=0.15,
                    help='the threshold to distinguish has/no watermark image frame')
parser.add_argument('--use_dictfile', action='store_true', default=False,
                    help='if true, use generated decoded watermarks file last time')
parser.add_argument('--no_accuracy', action='store_true', default=False,
                    help='if compute decoded accuracy')
parser.add_argument('--encode', action='store_true', default=False,
                    help='if use encoder to embedded watermark messages')
parser.add_argument('--decode', action='store_true', default=False,
                    help='if use decoder to extarct watermark messages')


class Water_Marker:

    def __init__(self, args, seed=0):
        assert os.path.exists(args.checkpoint)
        assert os.path.exists(args.input_videos_dir)
        assert os.path.exists(args.watermark_path)
        ### stable the random seed
        random.seed(seed)

        ### parameters define
        self.mark_index = 520
        self.video_idx  = 0
        self.videomark_label = {}
        self.final_decoded_watermark = {}

        ### accuracy parameters define
        self.correct_nums = 0
        self.total_nums = 0
        self.no_watermark_nums = 0

        ### parameters assignment
        self.checkpoint = args.checkpoint
        self.input_videos_dir = args.input_videos_dir
        self.output_dir = args.output_dir
        self.watermark_path = args.watermark_path
        self.label_path = args.label_path
        self.threshold = args.threshold
        self.use_dictfile = args.use_dictfile
        self.no_accuracy = args.no_accuracy
        self.encode = args.encode
        self.decode = args.decode

        if self.decode:
            assert os.path.exists(self.label_path)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        ### load model and checkpoints
        self.model = RivaGAN.load(self.checkpoint)
        #print(self.model)
        
        ### load original watermark messages
        self.water_marking = np.array(pickle.load(open(self.watermark_path, 'rb')))
        #print('--- water_marking_shape:', self.water_marking.shape)

    def generate_decoded_watermarks(self, decoded_messages):    
        final_decoded_watermark = {}
        for key, value in decoded_messages.items():
            value = np.array(value)
            biterr_index_list = []
            biterr_value_list = []
            for i in range(value.shape[0]):
                expand_value = value[i].repeat(self.water_marking.shape[0], axis = 0)
                bitwise_avg_err = np.sum(np.abs(expand_value-self.water_marking), \
                                         axis=1) / self.water_marking.shape[1]
            
                ### find min biterr between decoded message and water_marking in library(32bit-watermarks)
                currect_bitwise_err = np.min(bitwise_avg_err)
                currect_bitwise_err_index = np.argmin(bitwise_avg_err)
                biterr_value_list.append(currect_bitwise_err)
                biterr_index_list.append(currect_bitwise_err_index)
        
            biterr_index_array = np.array(biterr_index_list)
            biterr_value_array = np.array(biterr_value_list)

            ### find the most frequent watermark messages
            max_counts_watermark = np.argmax(np.bincount(biterr_index_array))
        
            ### determine whether the video is embedded with watermark by threshold
            if min(biterr_value_array) < self.threshold:
                self.final_decoded_watermark[key] = max_counts_watermark
            else:
                self.final_decoded_watermark[key] = 'no_mark'

    def encoder(self):
        ### define encoded videos and videomark label saved path
        video_saved_dir = os.path.join(self.output_dir, 'encoded_videos')
        videomark_label_path = os.path.join(self.output_dir, 'videomark_label.npy')
        if not os.path.exists(video_saved_dir):
            os.mkdir(video_saved_dir)

        for video in os.listdir(self.input_videos_dir):
            self.mark_index += 1
            #self.mark_index = random.randint(0, 49)
            self.video_idx  += 1
            messages = self.water_marking[self.mark_index]
            messages = torch.FloatTensor(messages).unsqueeze(0).cuda()

            #### define video input and output paths
            video_input_path = os.path.join(self.input_videos_dir, video)
            video_saved_path = os.path.join(video_saved_dir, video)

            ### add watermark to videos
            self.model.encode(video_input_path, messages, video_saved_path, self.video_idx)
            self.videomark_label[video] = self.mark_index
        
        ### saved videomark label
        np.save(videomark_label_path, self.videomark_label)
        print(f"=> 视频与数字水印关系对应表保存在: {videomark_label_path}")

    def decoder(self):
        ### load decoded watermark messages, if it exists.
        decoded_watermark_path = os.path.join(self.output_dir, 'decoded_watermarks.npy')
        decoded_watermark_dict = {}
        if self.use_dictfile and os.path.exists(decoded_watermark_path):
            decoded_watermark_dict = np.load(decoded_watermark_path, allow_pickle = True)
            decoded_watermark_dict = decoded_watermark_dict.tolist()
            print("=> Decoded watermark dictionary load successfully!!!")
        
        ### else, generate and saved decoded watermark messages
        else:
            print("=> Decoded watermark dictionary not exists, start generating...")
            for video in os.listdir(self.input_videos_dir):
                self.video_idx += 1
                video_path = os.path.join(self.input_videos_dir, video)

                ### extract watermark from videos
                decoded_watermarks = self.model.decode(video_path, self.video_idx, self.water_marking)
                decoded_watermark_dict[video] = decoded_watermarks
            
            ### decoded watermark messages saved
            np.save(decoded_watermark_path, decoded_watermark_dict)
        
        ### generate finally videos decoded watermark messages table
        self.generate_decoded_watermarks(decoded_watermark_dict)

        ### saved finally videos decoded watermark messages table
        result_file = os.path.join(self.output_dir, 'final_decoded_watermark.txt')
        ff = open(result_file, 'w')
        for k, v in self.final_decoded_watermark.items():
            ff.write(str(k) + ' ' + str(v) + '\n')
        ff.close()
        print('=> Final decoded watermark messages saved to {}'.format(result_file))

        ### compute decoded accuracy
        if not self.no_accuracy:
            self.decode_accuracy()

    def decode_accuracy(self):
        ### load videomark label
        if os.path.exists(self.label_path):
            videomark_label = np.load(self.label_path, allow_pickle = True)
        else:
            print('=> videomark label file not exists!!!')
        
        videomark_label = videomark_label.tolist()
        print('-' * 80)
        for video_name in os.listdir(self.input_videos_dir):
            self.total_nums += 1
            if self.final_decoded_watermark[video_name] == 'no_mark':
                self.no_watermark_nums += 1
                print('=> Video: [{video_name}] no watermark messasges!')
            elif self.final_decoded_watermark[video_name] == videomark_label[video_name]:
                self.correct_nums += 1
            else:
                print('=> Video: [{video_name}] has watermark messages, but decode error!')
        
        ### print informations(accuracy, decode correct nums, no watermark nums)
        if self.total_nums == self.no_watermark_nums:
            print("=> All videos have no watermark messages!!!")
        else:
            accuracy = self.correct_nums / (self.total_nums - self.no_watermark_nums)
            print("=> No watermark messages video nums: {}".format(self.no_watermark_nums))
            print("=> Decode correct_nums/has_watermark_nums: [{}/{}], accuracy: {}%".format(
                                                              self.correct_nums,
                                                              self.total_nums - self.no_watermark_nums,
                                                              accuracy*100.0))

   
if __name__ == "__main__":
    
    args = parser.parse_args()
    water_marker = Water_Marker(args)
    
    print("=" * 80)

    if args.encode:
        print('=> Run Encoder ...')
        water_marker.encoder()
    elif args.decode:
        print('=> Run Decoder ...')
        water_marker.decoder()
     
    print("=" * 80)
