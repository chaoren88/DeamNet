import argparse
import torch.nn as nn
from torch.autograd import Variable
import os
from DeamNet import Deam
from skimage.measure.simple_metrics import compare_psnr
import torch
import cv2
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description="AWGN Testing......")
parser.add_argument("--pretrained", type=str, default="./Deam_models/", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test dataset such as Set12, Set68 and Urban100')
parser.add_argument("--test_noiseL", type=float, default=15, help='noise level used on test set')

parser.add_argument('--data_dir', type=str, default='./Dataset')
parser.add_argument('--Isreal', default=False, help='If training/testing on RGB images')

opt = parser.parse_args()


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def normalize(data):
    return data/255.


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.numpy()
    return img


def main():
    print('Loading model ...\n')
    net = Deam(opt.Isreal)
    model = nn.DataParallel(net).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.pretrained, 'noise15.pth'), map_location=lambda storage, loc: storage))
    model.eval()

    # print('Loading data info ...\n')
    files_path = os.path.join(opt.data_dir, 'test', opt.test_data)
    files_source = os.listdir(files_path)

    psnr_test = 0
    i = 1

    for f in files_source:
        SEED = 0
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

        image_path = os.path.join(files_path, f)
        # image
        Img = cv2.imread(image_path)
        Img = normalize(np.float32(Img[:, :, 0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)

        # noise
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
        # noisy image
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())

        with torch.no_grad():  # this can save much memory
            B, C, H, W = INoisy.size()

            # padding to fit the input size of UNet
            bottom = (16 - H % 16) % 16
            right = (16 - H % 16) % 16

            padding = nn.ReflectionPad2d((0, right, 0, bottom))
            INoisy_input = padding(INoisy)

            model_out = model(INoisy_input)
            Out = model_out[:, :, 0:H, 0:W]

        psnr = batch_PSNR(torch.clamp(Out, 0., 1.), ISource, 1.)
        psnr_test += psnr
        i += 1
    psnr_test /= len(files_source)
    print("PSNR on test data %f" % psnr_test)
    print("\n")


if __name__ == "__main__":
    main()
