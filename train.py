import argparse
import os
import json
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from model import resnext50_32x4d
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

#默认使用命令：
#python ./train.py ./dataset   --save ./checkpoints     --log ./logs
#其中./dataset是数据集的路径,save是保存模型的路径，log是保存日志的路径

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains ResNeXt on CatsVsDogs', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('data_path', type=str, help='Root for the CatsVsDogs dataset.')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_bs', type=int, default=10)
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default='./', help='Log folder.')
    args = parser.parse_args()

    # Init logger
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')

    # Calculate number of epochs wrt batch size
    # args.epochs = args.epochs * 128 // args.batch_size
    args.schedule = [x * 128 // args.batch_size for x in args.schedule]

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)
        
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(256),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),transforms.Normalize(mean, std)])

    test_transform = transforms.Compose([
        transforms.Resize(256),transforms.CenterCrop(224),
        transforms.ToTensor(),transforms.Normalize(mean, std)])

    train_data = dset.ImageFolder(os.path.join(args.data_path, 'train'), train_transform)
    test_data = dset.ImageFolder(os.path.join(args.data_path, 'val'), test_transform)
    nlabels = 2  # cats vs dogs

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    # Init checkpoints
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    # Init model, criterion, and optimizer
    net = resnext50_32x4d(nlabels)
    #print(net)
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    # train function (forward, backward, update)
    def train():
        net.train()
        loss_avg = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())

            # forward
            output = net(data)

            # backward
            optimizer.zero_grad()
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # exponential moving average
            loss_avg = loss_avg * 0.2 + float(loss) * 0.8

        state['train_loss'] = loss_avg


    # test function (forward only)
    def test():
        net.eval()
        loss_avg = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())

            # test loss average
            loss_avg += float(loss)

        state['test_loss'] = loss_avg / len(test_loader)
        state['test_accuracy'] = correct / len(test_loader.dataset)


    # Main loop
    best_accuracy = 0.0
    for epoch in range(args.epochs):
        if epoch in args.schedule:
            state['learning_rate'] *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']

        state['epoch'] = epoch
        train()
        test()
        if state['test_accuracy'] > best_accuracy:
            best_accuracy = state['test_accuracy']
            torch.save(net.state_dict(), os.path.join(args.save, 'model.pytorch'))
        log.write('%s\n' % json.dumps(state))
        log.flush()
        print(state)
        print("Best accuracy: %f" % best_accuracy)

    log.close()

