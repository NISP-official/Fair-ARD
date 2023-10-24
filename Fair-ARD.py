from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
parser = argparse.ArgumentParser(description='Fair-ARD')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
parser.add_argument('--lr_factor', default=0.1, type=float, help='factor by which to decrease lr')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs for training')
parser.add_argument('--output', default = '', type=str, help='output subdirectory')
parser.add_argument('--model', default = 'ResNet18', type = str, help = 'student model name')
parser.add_argument('--teacher_model', default = 'WideResNet', type = str, help = 'teacher network model')
parser.add_argument('--teacher_path', default = '', type=str, help='path of teacher net being distilled')
parser.add_argument('--temp', default=30.0, type=float, help='temperature for distillation')
parser.add_argument('--val_period', default=1, type=int, help='print every __ epoch')
parser.add_argument('--save_period', default=1, type=int, help='save every __ epoch')
parser.add_argument('--alpha', default=1.0, type=float, help='weight for sum of losses')
parser.add_argument('--beta', default=2.0, type=float, help='beta for Fair-ARD')
args = parser.parse_args()
print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch in args.lr_schedule:
        lr *= args.lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
num_classes = len(testloader.dataset.classes)

class AttackPGD(nn.Module):
    def __init__(self, basic_net, config):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']

    def forward(self, inputs, targets):
        Kappa = [0 for _ in range(len(inputs))]
        x = inputs.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            student_out_pgd = self.basic_net(x)
            predict = student_out_pgd.max(1, keepdim=True)[1]
            # Update Kappa
            for p in range(len(x)):
                if predict[p] == targets[p]:
                    Kappa[p] += 1
            with torch.enable_grad():
                loss = F.cross_entropy(student_out_pgd, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0.0, 1.0)
        return self.basic_net(x), x, Kappa

print('==> Building model..'+args.model)
if args.model == 'MobileNetV2':
	basic_net = MobileNetV2(num_classes=num_classes)
elif args.model == 'WideResNet':
	basic_net = WideResNet(num_classes=num_classes)
elif args.model == 'ResNet18':
	basic_net = ResNet18(num_classes=num_classes)
basic_net = basic_net.to(device)
if args.teacher_path != '':
	if args.teacher_model == 'MobileNetV2':
		teacher_net = MobileNetV2(num_classes=num_classes)
	elif args.teacher_model == 'WideResNet':
		teacher_net = WideResNet(num_classes=num_classes)
	elif args.teacher_model == 'ResNet18':
		teacher_net = ResNet18(num_classes=num_classes)
	teacher_net = teacher_net.to(device)
	for param in teacher_net.parameters():
		param.requires_grad = False

config = {
    'epsilon': 8.0 / 255,
    'num_steps': 10,
    'step_size': 2.0 / 255,
}
net = AttackPGD(basic_net, config)
if device == 'cuda':
    cudnn.benchmark = True

print('==> Loading teacher..')
teacher_net.load_state_dict(torch.load(args.teacher_path))
teacher_net.eval()

KL_loss = nn.KLDivLoss()
XENT_loss = nn.CrossEntropyLoss()
lr=args.lr

def weight_assign(Kappa):
    for i in range(len(Kappa)):
        print('{:d}:{:.4f}  '.format(i, Kappa[i]), end='')
    print('')
    reweight = (1 / (Kappa + 1e-5)) ** args.beta
    sum_value = num_classes
    scale_factor = sum_value / torch.sum(reweight)
    reweight = reweight * scale_factor
    reweight = torch.clamp(reweight, min=5/6, max=5/2)
    return reweight

def train(epoch, optimizer, reweight):
    net.train()
    Kappa_total = [0 for _ in range(num_classes)]
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)     
        optimizer.zero_grad()
        outputs, pert_inputs, Kappa = net(inputs, targets)
        for i in range(len(Kappa)):
            Kappa_total[targets[i]] += Kappa[i]
        teacher_outputs = teacher_net(inputs)
        basic_outputs = basic_net(inputs)
        kl_loss_1_class = []
        ce_loss_class = []
        for i in range(len(outputs)):
            kl_loss_1_class.append(KL_loss(F.log_softmax(outputs[i]/args.temp, dim=0), F.softmax(teacher_outputs[i]/args.temp, dim=0)))
            ce_loss_class.append(XENT_loss(basic_outputs[i].unsqueeze(0), targets[i].unsqueeze(0)))
            if epoch >= 0:
                kl_loss_1_class[i] = kl_loss_1_class[i] * reweight[targets[i]]
                ce_loss_class[i] = ce_loss_class[i] * reweight[targets[i]]
        kl_loss_1 = (1/len(outputs)) * torch.sum(torch.stack(kl_loss_1_class))
        ce_loss = (1/len(outputs)) * torch.sum(torch.stack(ce_loss_class))
        loss = args.alpha*args.temp*args.temp*kl_loss_1+(1.0-args.alpha)*ce_loss
        loss.backward()
        optimizer.step()
    if (epoch+1)%args.save_period == 0:
        state = basic_net.state_dict()
        if not os.path.isdir('checkpoint_fair_ard/'+args.output+'/'):
            os.makedirs('checkpoint_fair_ard/'+args.output+'/', )
        torch.save(state, './checkpoint_fair_ard/'+args.output+'/epoch'+str(epoch)+'.pt')
    return Kappa_total

def main():
    lr = args.lr
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4, nesterov=True)
    reweight = torch.ones(num_classes)
    for epoch in range(args.epochs):
        print('epoch{:d}:'.format(epoch))
        adjust_learning_rate(optimizer, epoch, lr)
        Kappa_total = train(epoch, optimizer, reweight)
        Kappa_total = torch.tensor(Kappa_total) / (len(trainloader.dataset) / num_classes)
        reweight = weight_assign(Kappa_total)
        for i in range(len(reweight)):
            print('{:d}:{:.4f}  '.format(i, reweight[i]), end='')
        print('')
if __name__ == '__main__':
    main()
