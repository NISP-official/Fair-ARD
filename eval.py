import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from models import *

def cwloss(output, target,confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)
    loss = torch.sum(loss)
    return loss

def _pgd_whitebox(model,
                  X,
                  y,
                  device,
                  num_classes,
                  attack_method,
                  ):
    if attack_method == 'Clean':
        out = model(X)
        err = (out.data.max(1)[1] != y.data).float().sum()
        err_class = (out.data.max(1)[1] != y.data).float()
        err_class_num = [0 for x in range(num_classes)]
        for i in range(len(y.data)):
            err_class_num[y.data[i]] += err_class[i].cpu().numpy()
        return err, err_class_num
    if attack_method == 'FGSM':
        epsilon=8. / 255.
        num_steps=1
        step_size=8. / 255.
    elif attack_method == 'PGD' or attack_method == 'C&W':
        epsilon=8. / 255.
        num_steps=20
        step_size=2. / 255.
    
    X_pgd = Variable(X.data, requires_grad=True)

    if attack_method == 'FGSM':
        random_noise = 0.001 * torch.randn(*X_pgd.shape).cuda().detach()
    else:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            if attack_method == 'C&W':
                loss = cwloss(model(X_pgd), y, num_classes=num_classes)
            else:
                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    out_pgd = model(X_pgd)
    err_pgd = (out_pgd.data.max(1)[1] != y.data).float().sum()
    err_pgd_class = (out_pgd.data.max(1)[1] != y.data).float()
    err_pgd_class_num = [0 for i in range(num_classes)]
    for i in range(len(y.data)):
        err_pgd_class_num[y.data[i]] += err_pgd_class[i].cpu().numpy()
    return err_pgd, err_pgd_class_num

def eval_adv_test_whitebox(model, device, test_loader, attack_method):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    num_classes = len(test_loader.dataset.classes)
    robust_err_class_total = np.zeros(num_classes)
    target_class = np.zeros(num_classes)
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust, err_robust_class = _pgd_whitebox(model, X, y, device, num_classes, attack_method)
        robust_err_total += err_robust
        for i in range(len(y)):
            target_class[y[i]] += 1
        for i in range(num_classes):
            robust_err_class_total[i] += err_robust_class[i]
    for i in range(num_classes):
        print('class{:2d}: Robust: ({:.2f}%)'.format(i, 100. - 100. * robust_err_class_total[i] / target_class[i]))
    robustness = 1 - robust_err_total / len(test_loader.dataset)
    print('Robust: ({:.2f}%)'.format(100 * robustness))
    robustness_values = 1 - robust_err_class_total / target_class
    worst_class = np.argmin(robustness_values)
    worst_robustness = robustness_values[worst_class]
    print('Worst{:d}: ({:.2f}%)'.format(worst_class, 100 * worst_robustness))
    return robustness, worst_robustness

def evaluate_attack(model, device, test_loader, atk):
    rob = 0
    model.eval()
    num_classes = len(test_loader.dataset.classes)
    rob_class = np.zeros(num_classes)
    target_class = np.zeros(num_classes)
    for data, target in test_loader:
        X, y = data.to(device), target.to(device)

        X_adv = atk(X, y)

        with torch.no_grad():
            output = model(X_adv)
        rob += (output.max(1)[1] == y).sum().item()
        test_acc_class_temp = (output.max(1)[1] == y).float()
        for j in range(y.size(0)):
            rob_class[y[j]] += test_acc_class_temp[j].cpu().numpy()
            target_class[y[j]] += 1

    for i in range(num_classes):
        print('class{:2d}: Robust: ({:.2f}%)'.format(i, 100. * rob_class[i] / target_class[i]))
    robustness = rob / len(test_loader.dataset)
    print('Robust: ({:.2f}%)'.format(100 * robustness))
    robustness_values = rob_class / target_class
    worst_class = np.argmin(robustness_values)
    worst_robustness = robustness_values[worst_class]
    print('Worst{:d}: ({:.2f}%)'.format(worst_class, 100 * worst_robustness))
    return robustness, worst_robustness

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8./255.)
    parser.add_argument('--model', type=str, default='ResNet18')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--whitebox', type=bool, default=True)
    parser.add_argument('--aa', type=bool, default=False)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    if 'cifar100' not in args.model:
        item = datasets.CIFAR10(root='../data/cifar10', train=False, transform=transform_chain, download=True)
        test_loader = data.DataLoader(item, batch_size=args.batch_size, shuffle=False, num_workers=0)
    elif 'cifar100' in args.model:
        item = datasets.CIFAR100(root='../data/cifar100', train=False, transform=transform_chain, download=True)
        test_loader = data.DataLoader(item, batch_size=args.batch_size, shuffle=False, num_workers=0)
    num_classes = len(test_loader.dataset.classes)
    # load model
    model_path = args.model_path
    print(model_path)
    if args.model == 'ResNet18':
        model = ResNet18(num_classes=num_classes)
    elif args.model == 'MobileNetV2':
        model = MobileNetV2(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    
    robustness_list = []
    worst_robustness_list = []

    # white-box attack
    if args.whitebox:
        attack_method_list = ['Clean', 'FGSM', 'PGD', 'C&W']
        for i, attack_method in enumerate(attack_method_list):
            print(attack_method)
            robustness, worst_robustness = eval_adv_test_whitebox(model, device, test_loader, attack_method)
            robustness_list.append(robustness)
            worst_robustness_list.append(worst_robustness)

    # load attack
    if args.aa:     
        import torchattacks
        print('autoattack')
        atk = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=num_classes)
        robustness, worst_robustness = evaluate_attack(model, device, test_loader, atk)
        robustness_list.append(robustness)
        worst_robustness_list.append(worst_robustness)
    
    for i in range(len(robustness_list)):
        print('{:.4f}\t{:.4f}\t'.format(robustness_list[i], worst_robustness_list[i]), end='')
    print()
                
