import os
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
from models import *
import torchvision
from torchvision import transforms

parser = argparse.ArgumentParser(description='Fair-RSLAD')
parser.add_argument('--teacher_path', default = '', type=str, help='path of teacher net being distilled')
parser.add_argument('--beta', default=2.0, type=float, help='beta for Fair-RSLAD')
args = parser.parse_args()
print(args)

# we fix the random seed to 0, this method can keep the results consistent in the same conputer.
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
prefix = 'resnet18-CIFAR10_RSLAD'
epochs = 300
batch_size = 128
epsilon = 8/255.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

def rslad_inner_loss(model,
                teacher_logits,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False,reduce=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    Kappa = [0 for _ in range(len(x_adv))]
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        student_out_pgd = model(x_adv)
        predict = student_out_pgd.max(1, keepdim=True)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == y[p]:
                Kappa[p] += 1
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(student_out_pgd, dim=1),
                                       F.softmax(teacher_logits, dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    logits = model(x_adv)
    return logits, Kappa

student = ResNet18()
student = torch.nn.DataParallel(student)
student = student.cuda()
student.train()
optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4, nesterov=True)
def kl_loss(a,b):
    loss = -a*b + torch.log(b+1e-5)*b
    return loss
teacher = WideResNet()
teacher.load_state_dict(torch.load(args.teacher_path))
teacher = torch.nn.DataParallel(teacher)
teacher = teacher.cuda()
teacher.eval()
num_classes = len(testloader.dataset.classes)
def weight_assign(Kappa):
    for i in range(len(Kappa)):
        print('{:d}:{:.4f}  '.format(i, Kappa[i]), end='')
    print('')
    reweight = (1 / Kappa) ** args.beta
    sum_value = num_classes
    scale_factor = sum_value / torch.sum(reweight)
    reweight = reweight * scale_factor
    reweight = torch.clamp(reweight, min=5/6, max=5/2)
    return reweight
model_dir = './checkpoint_fair_rslad'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
robustness = []
reweight = torch.ones(10)
for epoch in range(1,epochs+1):
    print('epoch{:d}:'.format(epoch))
    Kappa_total = [0 for _ in range(10)]
    for step,(train_batch_data,train_batch_labels) in enumerate(trainloader):
        student.train()
        train_batch_data = train_batch_data.float().cuda()
        train_batch_labels = train_batch_labels.cuda()
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher(train_batch_data)

        adv_logits, Kappa = rslad_inner_loss(student,teacher_logits,train_batch_data,train_batch_labels,optimizer,step_size=2/255.0,epsilon=epsilon,perturb_steps=10)
        for i in range(len(Kappa)):
            Kappa_total[train_batch_labels[i]] += Kappa[i]
        student.train()
        nat_logits = student(train_batch_data)
        kl_Loss1_class = []
        kl_Loss2_class = []
        for i in range(len(train_batch_labels)):
            kl_Loss1_class.append(kl_loss(F.log_softmax(adv_logits[i]), F.softmax(teacher_logits.detach()[i])))
            kl_Loss2_class.append(kl_loss(F.log_softmax(nat_logits[i]), F.softmax(teacher_logits.detach()[i])))
            if epoch > 0:
                kl_Loss1_class[i] *= reweight[train_batch_labels[i]]
                kl_Loss2_class[i] *= reweight[train_batch_labels[i]]
        kl_Loss1 = (1.0 / len(train_batch_labels)) * sum(kl_Loss1_class)
        kl_Loss2 = (1.0 / len(train_batch_labels)) * sum(kl_Loss2_class)
        kl_Loss1 = torch.mean(kl_Loss1)
        kl_Loss2 = torch.mean(kl_Loss2)
        loss = 5/6.0*kl_Loss1 + 1/6.0*kl_Loss2
        loss.backward()
        optimizer.step()
    Kappa_total = torch.tensor(Kappa_total) / (len(trainloader.dataset) / num_classes)
    reweight = weight_assign(Kappa_total)
    for i in range(len(reweight)):
        print('{:d}:{:.4f}  '.format(i, reweight[i]), end='')
    print('')

    if epoch in [215,260,285]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    torch.save(student.state_dict(),
                       os.path.join(model_dir, 'epoch{}.pt'.format(epoch)))