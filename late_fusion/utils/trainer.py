import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from copy import deepcopy

def train(epoch, model, optimizer, cri_cls, train_loader, gpu, args):
    loss_cls_ = 0.
    loss_reg_ = 0.
    acc_tr = 0.

    # Training
    model.train()
    for ix, (tr_x, tr_aux, tr_y) in enumerate(train_loader):
        # GPU Implementation
        if gpu == True:
            tr_x = tr_x.cuda()
            tr_aux = tr_aux.cuda()
            tr_y = tr_y.cuda().view(-1)
        else:
            tr_x = tr_x.cpu()
            tr_aux = tr_aux.cpu()
            tr_y = tr_y.cpu().view(-1)

        # Foward Propagation
        tr_aux = tr_aux.view(tr_aux.size()[0], 1, -1)
        out_cls = model(tr_x, tr_aux)

        loss_cls = cri_cls(out_cls, tr_y)
        loss_tr = loss_cls

        # BackPropagation
        optimizer.zero_grad()
        loss_tr.backward()
        optimizer.step()

        # Report
        loss_cls_ += loss_cls.data

        _, predicted_l = torch.max(out_cls.data, dim=1)
        predicted_l = predicted_l.view(-1)
        target_l = tr_y.data
        acc_tr += (torch.mean((predicted_l == target_l).float())) * 100

    acc_tr = acc_tr / len(train_loader)
    loss_cls_ = loss_cls_ / len(train_loader)

    if epoch % args.print_epoch == 0:
        if args.print_test:
            print('Epoch : %d, Train Acc : %2.2f, Train Loss_cls : %.2f' % (epoch, acc_tr, loss_cls_))

    return model

def test(epoch, model, cri_cls, test_loader, gpu, args):
    model.eval()
    acc_te_ = 0.
    loss_cls_ = 0.

    confusion_matrix = np.zeros([3, 3])

    result_dict = {}
    with torch.no_grad():
        for ix, (te_x, te_aux, te_y) in enumerate(test_loader):
            # GPU Implementation
            if gpu == True:
                te_x = te_x.cuda()
                te_aux = te_aux.cuda()
                te_y = te_y.cuda().view(-1)
            else:
                te_x = te_x.cpu()
                te_aux = te_aux.cpu()
                te_y = te_y.cpu().view(-1)

            # Foward Propagation
            te_aux = te_aux.view(te_aux.size()[0], 1, -1)
            out_cls = model(te_x, te_aux)

            loss_cls = cri_cls(out_cls, te_y)

            # Accuracy
            _, pred_l = torch.max(out_cls, dim=1)
            target_l = te_y.data
            acc_te_ += (torch.mean((pred_l == target_l).float())).data * 100

            f1 = {'macro':0., 'micro':0., 'weighted':0.}
            f1['macro'] += f1_score(target_l.cpu().numpy(), pred_l.cpu().numpy(), average='macro')
            f1['micro'] += f1_score(target_l.cpu().numpy(), pred_l.cpu().numpy(), average='micro')
            f1['weighted'] += f1_score(target_l.cpu().numpy(), pred_l.cpu().numpy(), average='weighted')

            loss_cls_ += loss_cls.data
            for true_label, predicted_label in zip(target_l, pred_l):
                confusion_matrix[true_label, predicted_label] += 1

        # Loss
        loss_cls_ /= len(test_loader)
        f1['macro'] /= len(test_loader)
        f1['micro'] /= len(test_loader)
        f1['weighted'] /= len(test_loader)
        acc_te_ /= len(test_loader)

    result_dict['acc'] = acc_te_
    result_dict['confusion'] = confusion_matrix
    result_dict['f1'] = f1
    result_dict['loss'] = loss_cls_
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
