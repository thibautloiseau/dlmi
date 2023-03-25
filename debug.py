import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

# from models.vision_transformer import SwinUnet
from models.unet import UNet
# from config import get_config
from data import RadDataset

# parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str,
#                     default='../data/Synapse/train_npz', help='root dir for data')
# parser.add_argument('--dataset', type=str,
#                     default='Synapse', help='experiment_name')
# parser.add_argument('--list_dir', type=str,
#                     default='./lists/lists_Synapse', help='list dir')
# parser.add_argument('--num_classes', type=int,
#                     default=9, help='output channel of network')
# parser.add_argument('--output_dir', type=str, help='output dir')
# parser.add_argument('--max_iterations', type=int,
#                     default=30000, help='maximum epoch number to train')
# parser.add_argument('--max_epochs', type=int,
#                     default=150, help='maximum epoch number to train')
# parser.add_argument('--batch_size', type=int,
#                     default=24, help='batch_size per gpu')
# parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
# parser.add_argument('--deterministic', type=int,  default=1,
#                     help='whether use deterministic training')
# parser.add_argument('--base_lr', type=float,  default=0.01,
#                     help='segmentation network learning rate')
# parser.add_argument('--img_size', type=int,
#                     default=224, help='input patch size of network input')
# parser.add_argument('--seed', type=int,
#                     default=1234, help='random seed')
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
# parser.add_argument(
#         "--opts",
#         help="Modify config options by adding 'KEY VALUE' pairs. ",
#         default=None,
#         nargs='+',
#     )
# parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
# parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
#                     help='no: no cache, '
#                             'full: cache all data, '
#                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
# parser.add_argument('--resume', help='resume from checkpoint')
# parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
# parser.add_argument('--use-checkpoint', action='store_true',
#                     help="whether to use gradient checkpointing to save memory")
# parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
#                     help='mixed precision opt level, if O0, no amp is used')
# parser.add_argument('--tag', help='tag of experiment')
# parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
# parser.add_argument('--throughput', action='store_true', help='Test throughput only')
#
# args = parser.parse_args()
# if args.dataset == "Synapse":
#     args.root_path = os.path.join(args.root_path, "train_npz")
# config = get_config(args)


if __name__ == "__main__":

    # Trying to load pretrained Swin-UNet but too big for GPU...
    if False:
        if not args.deterministic:
            cudnn.benchmark = True
            cudnn.deterministic = False
        else:
            cudnn.benchmark = False
            cudnn.deterministic = True

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        if args.batch_size != 24 and args.batch_size % 6 == 0:
            args.base_lr *= args.batch_size / 24

        cuda = True if torch.cuda.is_available() else False
        print(f"Using cuda device: {cuda}")  # check if GPU is used

        # Tensor type (put everything on GPU if possible)
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # Creating network from config and adapting it to out input and output
        net = SwinUnet(config, img_size=args.img_size).cuda()
        net.load_from(config)

        print(sum([p.numel() for p in net.parameters()]))

        # Creating dataset
        dataset = RadDataset('train')

        for sample in dataset:
            ct = sample["ct"].type(Tensor)
            possible_dose_mask = sample["possible_dose_mask"].type(Tensor)
            structure_masks = sample["structure_masks"].type(Tensor)
            dose = sample["dose"].type(Tensor)

            y_pred = net(T.Resize(224)(ct.type(Tensor)[None, :]))

    # Pretrained UNet
    if False:
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)

        cuda = True if torch.cuda.is_available() else False
        print(f"Using cuda device: {cuda}")  # check if GPU is used

        if cuda:
            model = model.cuda()

        # Tensor type (put everything on GPU if possible)
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # Creating dataset
        dataset = RadDataset('train')

        for sample in tqdm(dataset):
            ct = sample["ct"].type(Tensor)
            possible_dose_mask = sample["possible_dose_mask"].type(Tensor)
            structure_masks = sample["structure_masks"].type(Tensor)
            dose = sample["dose"].type(Tensor)

            ct = T.Resize(256)(ct.repeat(3, 1, 1)[None, :])
            print(ct.shape)

            plt.imshow(ct.squeeze()[0].cpu().numpy())
            plt.show()

            y_pred = model(ct)
            plt.imshow(y_pred.detach().cpu().numpy().squeeze()[:, :, None])
            plt.show()

            print(y_pred.shape)

    # Visualizing predictions
    if True:
        # Loading trained model
        model = UNet()
        model.load_state_dict(torch.load('important_logs/with_aug/model.pt'))

        cuda = True if torch.cuda.is_available() else False
        print(f"Using cuda device: {cuda}")  # check if GPU is used

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        if cuda:
            model.cuda()

        dataset = RadDataset('train', transform=None)
        sample = dataset[0]

        ct, dose, possible_dose_mask, structure_masks = sample['ct'].type(Tensor), sample['dose'].type(Tensor), sample['possible_dose_mask'].type(Tensor), sample['structure_masks'].type(Tensor)

        x = torch.cat([ct, structure_masks], 0)[None, :]
        y_pred = model(x, possible_dose_mask).squeeze()
        print(y_pred.shape)

        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        plt.imshow(y_pred.detach().cpu()[:, :, None], cmap='jet')
        plt.show()

        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        plt.imshow(dose.detach().cpu().permute(1, 2, 0), cmap='jet')
        plt.show()


