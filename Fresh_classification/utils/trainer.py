import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from copy import deepcopy

# Seed
torch.manual_seed(41)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(41)

def train(epoch, model, optimizer, criterion, train_loader, option, scheduler=None):
    if scheduler is not None:
        scheduler.step(epoch)

    loss_center_ = 0.
    loss_cls_ = 0.
    acc_tr = 0.

    cri_cls = criterion['cls']
    cri_center = criterion['center']

    optim_cls = optimizer['cls']
    optim_center = optimizer['center']

    # Aux_calculate
    if option['aux'] is not None:
        model.eval()
        with torch.no_grad():
            aux_output = model(option['aux'])
    else:
        aux_output = None

    # Training
    for ix, (tr_x, tr_y) in enumerate(train_loader):
        model.train()
        # GPU Implementation
        if option['gpu'] == True:
            tr_x = tr_x.cuda()
            tr_y = tr_y.cuda().view(-1)

        else:
            tr_x = tr_x.cpu()
            tr_y = tr_y.cpu().view(-1)

        # Foward Propagation
        output = model(tr_x)

        loss_cls = cri_cls(output, tr_y)
        loss_center = cri_center(output, tr_y, aux_output) * option['alpha']

        loss_tr = loss_cls + loss_center

        # BackPropagation
        optim_cls.zero_grad()
        optim_center.zero_grad()

        loss_tr.backward()

        optim_cls.step()
        for param in cri_center.parameters():
            param.grad.data *= (1. / (option['alpha']+1e-4))
        optim_center.step()

        # Report
        loss_center_ += loss_center.data
        loss_cls_ += loss_cls.data
        _, predicted_l = torch.max(output.data, dim=1)
        predicted_l = predicted_l.view(-1)
        target_l = tr_y.data
        acc_tr += (torch.mean((predicted_l == target_l).float())) * 100

    acc_tr = acc_tr / len(train_loader)
    loss_center_ = loss_center_ / len(train_loader)
    loss_cls_ = loss_cls_ / len(train_loader)


    cri = loss_center_ + loss_cls_
    if cri > 1000:
        exit()

    if epoch % option['print_epoch'] == 0:
        if option['print']:
            print('Epoch : %d, Train Acc : %2.2f, Train Loss_center : %.2f, Train Loss_cls : %.2f' % (epoch, acc_tr, loss_center_, loss_cls_))
    return model

def test(epoch, model, criterion, test_loader, option, num_class=2):
    model.eval()
    acc_te = 0.

    confusion_matrix = np.zeros([num_class, num_class])

    cri_cls = criterion['cls']

    result_dict = {}
    with torch.no_grad():
        for ix, (te_x, te_y) in enumerate(test_loader):
            # GPU Implementation
            if option['gpu'] == True:
                te_x = te_x.cuda()
                te_y = te_y.cuda().view(-1)
            else:
                te_x = te_x.cpu()
                te_y = te_y.cpu().view(-1)

            # Foward Propagation
            output = model(te_x)

            # Accuracy
            _, pred_l = torch.max(output, dim=1)
            target_l = te_y.data
            acc_te += (torch.mean((pred_l == target_l).float())) * 100

            f1 = {}
            f1['macro'] = f1_score(target_l.cpu().numpy(), pred_l.cpu().numpy(), average='macro')
            f1['micro'] = f1_score(target_l.cpu().numpy(), pred_l.cpu().numpy(), average='micro')
            f1['weighted'] = f1_score(target_l.cpu().numpy(), pred_l.cpu().numpy(), average='weighted')

            for true_label, predicted_label in zip(target_l, pred_l):
                confusion_matrix[true_label, predicted_label] += 1
            # Loss
            loss_cls = cri_cls(output, te_y).data

    result_dict['acc'] = acc_te
    result_dict['confusion'] = confusion_matrix
    result_dict['f1'] = f1
    result_dict['loss'] = loss_cls
    return result_dict

def ad_train(epoch, model, optimizer, criterion, train_loader, option):
    model.train()

    loss_ = 0.
    acc_tr = 0.

    optim_cls = optimizer['cls']
    cri_cls = criterion['cls']

    for ix, (tr_x, tr_y) in enumerate(train_loader):
        # GPU Implementation
        if option['gpu'] == True:
            tr_x = tr_x.cuda()
            tr_y = tr_y.cuda().view(-1)

        else:
            tr_x = tr_x.cpu()
            tr_y = tr_y.cpu().view(-1)

        # Foward Propagation
        output = model(tr_x)
        loss_tr = cri_cls(output, tr_y)

        # BackPropagation
        optim_cls.zero_grad()
        loss_tr.backward()
        optim_cls.step()

        # Report
        loss_ += loss_tr.data
        _, predicted_l = torch.max(output.data, dim=1)
        predicted_l = predicted_l.view(-1)
        target_l = tr_y.data
        acc_tr += (torch.mean((predicted_l == target_l).float())) * 100

    acc_tr = acc_tr / len(train_loader)
    loss_ = loss_ / len(train_loader)

    if epoch % option['print_epoch'] == 0:
        if option['print']:
            print('\n Epoch : %d, Train Acc : %2.2f, train_loss : %.2f' % (epoch, acc_tr, loss_))
    return model
