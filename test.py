import argparse
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from model import resnext50_32x4d
import os
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

# 默认使用命令：
# python ./test.py ./dataset   --load ./checkpoints/model.pytorch
# 其中./dataset是数据集的路径，后面是选择要使用的训练好的模型

def get_args():
    parser = argparse.ArgumentParser(description='Test ResNeXt on CatsVsDogs', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('data_path', type=str, help='Root for the CatsVsDogs dataset.')
    # Optimization options
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size.')
    parser.add_argument('--test_bs', type=int, default=10)
    # Checkpoints
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default='./', help='Log folder.')
    args = parser.parse_args()
    return args

def test():
    # define default variables
    args = get_args()# divide args part and call it as function
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    state = {k: v for k, v in args._get_kwargs()}

    # prepare test data parts
    test_transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
            transforms.ToTensor(),transforms.Normalize(mean, std)])
    test_data = dset.ImageFolder(os.path.join(args.data_path, 'val'), test_transform)
    nlabels = 2

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    # initialize model and load from checkpoint
    net = resnext50_32x4d(nlabels)
    loaded_state_dict = torch.load(args.load)
    temp = {}
    for key, val in list(loaded_state_dict.items()):
        if 'module' in key:
            # parsing keys for ignoring 'module.' in keys
            temp[key[7:]] = val
        else:
            temp[key] = val
    loaded_state_dict = temp
    net.load_state_dict(loaded_state_dict)

    # paralleize model 
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    if args.ngpu > 0:
        net.cuda()
   
    # use network for evaluation 
    net.eval()

    # calculation part
    loss_avg = 0.0
    correct = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum()

            # test loss average
            loss_avg += loss.item()

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)

    # finally print state dictionary
    print(state)

if __name__=='__main__':
    test()
