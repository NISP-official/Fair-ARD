import os
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
from models import *
import torchvision
from torchvision import transforms
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='Fair-MTARD')
parser.add_argument('--adv_teacher_path', default = '', type=str, help='path of teacher net being distilled')
parser.add_argument('--nat_teacher_path', default = '', type=str, help='path of teacher net being distilled')
parser.add_argument('--beta', default=2.0, type=float, help='beta for Fair-MTARD')
args = parser.parse_args()
print(args)

prefix = 'resnet18_CIFAR10_MTRAD_'
epochs = 300
batch_size = 128
epsilon = 8/255.0
weight_learn_rate = 0.025
bert = 1

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

def mtard_inner_loss_ce(model,
                teacher_adv_model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0):

    criterion_ce_loss = torch.nn.CrossEntropyLoss().cuda()
    model.eval()
    batch_size = len(x_natural)
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    Kappa = [0 for _ in range(len(x_natural))]
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        logits_adv = model(x_adv)
        predict = logits_adv.max(1, keepdim=True)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == y[p]:
                Kappa[p] += 1
        with torch.enable_grad():
            loss_ce = criterion_ce_loss(logits_adv, y.cuda())
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    student_logits = model(x_adv)
    with torch.no_grad():
        teacher_logits = teacher_adv_model(x_adv)
    return student_logits, teacher_logits, Kappa

student = ResNet18()
student = student.cuda()
student.train()
optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4, nesterov=True)

weight = {
    "adv_loss": 1/2.0,
    "nat_loss": 1/2.0,
}
init_loss_nat = None
init_loss_adv = None

def kl_loss(a,b):
    loss = -a*b + torch.log(b+1e-5)*b
    return loss
teacher = WideResNet()
teacher.load_state_dict(torch.load(args.adv_teacher_path))
teacher = teacher.cuda()
teacher.eval()

teacher_nat = resnet56()
teacher_nat.load_state_dict(torch.load(args.nat_teacher_path))
teacher_nat = teacher_nat.cuda()
teacher_nat.eval()

model_dir = './checkpoint_fair_mtard'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
robustness = []
device = torch.device("cuda" if True else "cpu")
num_classes = len(testloader.dataset.classes)
reweight = torch.ones(num_classes)

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

for epoch in range(1,epochs+1):
    print('the {}th epoch '.format(epoch))
    Kappa_total = [0 for _ in range(num_classes)]
    for step,(train_batch_data,train_batch_labels) in enumerate(trainloader):
        student.train()
        train_batch_data = train_batch_data.float().cuda()
        train_batch_labels = train_batch_labels.cuda()
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_nat_logits = teacher_nat(train_batch_data)
        student_adv_logits,teacher_adv_logits, Kappa = mtard_inner_loss_ce(student,teacher,train_batch_data,train_batch_labels,optimizer,step_size=2/255.0,epsilon=epsilon,perturb_steps=10)
        for i in range(len(Kappa)):
            Kappa_total[train_batch_labels[i]] += Kappa[i]
        student.train()
        student_nat_logits = student(train_batch_data)
        kl_Loss1_class = []
        kl_Loss2_class = []
        for i in range(len(student_adv_logits)):
            kl_Loss1_class.append(kl_loss(F.log_softmax(student_adv_logits[i], dim=0), F.softmax(teacher_adv_logits[i], dim=0)))
            kl_Loss2_class.append(kl_loss(F.log_softmax(student_nat_logits[i], dim=0), F.softmax(teacher_adv_logits[i], dim=0)))
            if epoch >= 0:
                kl_Loss1_class[i] = kl_Loss1_class[i] * reweight[train_batch_labels[i]]
                kl_Loss2_class[i] = kl_Loss2_class[i] * reweight[train_batch_labels[i]]
        kl_Loss1 = (1/len(student_adv_logits)) * sum(kl_Loss1_class)
        kl_Loss2 = (1/len(student_adv_logits)) * sum(kl_Loss2_class)
        kl_Loss1 = torch.mean(kl_Loss1)
        kl_Loss2 = torch.mean(kl_Loss2)

        if init_loss_nat == None:
            init_loss_nat = kl_Loss2.item()
        if init_loss_adv == None:
            init_loss_adv = kl_Loss1.item()

        lhat_adv = kl_Loss1.item() / init_loss_adv
        lhat_nat = kl_Loss2.item() / init_loss_nat


        inv_rate_adv = lhat_adv**bert
        inv_rate_nat = lhat_nat**bert


        #weight_learn_rate = 0.025
        weight["nat_loss"] = weight["nat_loss"] - weight_learn_rate *(weight["nat_loss"] - inv_rate_nat/(inv_rate_adv + inv_rate_nat))
        #weight["adv_loss"] = weight["adv_loss"] - weight_learn_rate *(weight["adv_loss"] - inv_rate_adv/(inv_rate_adv + inv_rate_nat))
        weight["adv_loss"] = 1 - weight["nat_loss"] 

        total_loss = weight["adv_loss"]*kl_Loss1 + weight["nat_loss"]*kl_Loss2

        total_loss.backward()
        optimizer.step()

        if step%100 == 0:
            print('weight_nat: ', weight["nat_loss"],'nat_loss: ',kl_Loss2.item(),' weight_adv: ', weight["adv_loss"],' adv_loss: ',kl_Loss1.item())
    Kappa_total = torch.tensor(Kappa_total) / (len(trainloader.dataset) / num_classes)
    reweight = weight_assign(Kappa_total)
    for i in range(len(reweight)):
        print('{:d}:{:.4f}  '.format(i, reweight[i]), end='')
    print('')
    torch.save(student.state_dict(),
                       os.path.join(model_dir, 'epoch{}.pt'.format(epoch)))
    if epoch in [215,260,285]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        weight_learn_rate *= 0.1
