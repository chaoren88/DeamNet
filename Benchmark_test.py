from real import denoise_srgb, bundle_submissions_srgb, SIDD_denoise
import torch
from DeamNet import Deam
import os
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=str, default='./Deam_models/', help="Checkpoints directory,  (default:./checkpoints)")
parser.add_argument('--Isreal', default=True, help='Location to save checkpoint models')
parser.add_argument('--data_folder', type=str, default='./Dataset/Benchmark_test', help='Location to save checkpoint models')
parser.add_argument('--out_folder', type=str, default='./Dnd_result', help='Location to save checkpoint models')
parser.add_argument('--model', type=str, default='Real.pth', help='Location to save checkpoint models')
parser.add_argument('--Type', type=str, default='SIDD', help='To choose the testing benchmark dataset, SIDD or Dnd')
args = parser.parse_args()
use_gpu = True

print('Loading the Model')
net = Deam(args.Isreal)
checkpoint = torch.load(os.path.join(args.pretrained, args.model))
model = torch.nn.DataParallel(net).cuda()
model.eval()

if args.Type == 'Dnd':
    denoise_srgb(model, args.data_folder, args.out_folder)
    bundle_submissions_srgb(args.out_folder)
else:
    SIDD_denoise.test(args)

