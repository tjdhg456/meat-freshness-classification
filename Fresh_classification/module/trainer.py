import numpy as np
import torch
from tqdm import tqdm

def train(model, optimizer, epoch, train_loader, option, criterion, scheduler=None):
    model.train()

    if scheduler is not None:
        scheduler.step(epoch)

    loss_ = 0.
    acc_tr = 0.

    cri_reg = criterion[0]
    cri_cls = criterion[1]

    for ix, (tr_x, tr_reg, tr_cls) in enumerate(train_loader):
        # GPU Implementation
        if option['gpu'] == True:
            tr_x = tr_x.cuda()
            tr_reg = tr_reg.cuda().view(-1)
            tr_cls = tr_cls.cuda().view(-1)
        else:
            tr_x = tr_x.cpu()
            tr_reg = tr_reg.cpu().view(-1)
            tr_cls = tr_cls.cpu().view(-1)

        # Foward Propagation
        if option['type_name'] == 'reg':
            pred_reg = model(tr_x)
            loss_tr = cri_reg(pred_reg.view(-1), tr_reg)

        elif option['type_name'] == 'cls':
            pred_cls = model(tr_x)
            loss_tr = cri_cls(pred_cls, tr_cls)

        else:
            pred_reg, pred_cls = model(tr_x)
            loss_reg = cri_reg(pred_reg.view(-1), tr_reg)
            loss_cls = cri_cls(pred_cls, tr_cls)
            loss_tr = loss_reg * 0.3 + loss_cls

        if option['reg'] == True:
            l1_regularization, l2_regularization = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
            for param in model.parameters():
                l1_regularization += option['lambda1'] * torch.norm(param, 1)
                l2_regularization += option['lambda2'] * torch.norm(param, 2)

            loss_tr = loss_tr + l1_regularization + l2_regularization

        # BackPropagation
        optimizer.zero_grad()
        loss_tr.backward()
        optimizer.step()

        loss_ += loss_tr.data

        if option['type_name'] == 'reg':
            pass
        else:
            _, predicted_l = torch.max(pred_cls.data, dim=1)
            predicted_l = predicted_l.view(-1)
            target_l = tr_cls.data
            acc_tr += (torch.mean((predicted_l == target_l).float())) * 100

    if option['type_name'] == 'reg':
        acc_tr = -1
    else:
        acc_tr = acc_tr / len(train_loader)

    loss_ = loss_ / len(train_loader)

    if (epoch+1) % option['print_epoch'] == 0:
        print('Epoch : %d, Train Acc : %2.2f, Train Loss : %.2f,' % (epoch, acc_tr, loss_))

    return model

def test(model, epoch, test_loader, option, criterion):
    model.eval()
    acc_te = 0.

    cri_reg = criterion[0]

    if option['type_name'] != 'reg':
        confusion_matrix = np.zeros([option['num_class'], option['num_class']])

    with torch.no_grad():
        for ix, (te_x, te_reg, te_cls) in enumerate(test_loader):
            # GPU Implementation
            if option['gpu'] == True:
                te_x = te_x.cuda()
                te_reg = te_reg.cuda().view(-1)
                te_cls = te_cls.cuda().view(-1)
            else:
                te_x = te_x.cpu()
                te_reg = te_reg.cpu().view(-1)
                te_cls = te_cls.cpu().view(-1)

            # Foward Propagation
            if option['type_name'] == 'reg':
                pred_reg = model(te_x)
                loss_te = cri_reg(pred_reg.view(-1), te_reg)

            elif option['type_name'] == 'cls':
                pred_cls = model(te_x)

            else:
                _, pred_cls = model(te_x)

            # Accuracy
            if option['type_name'] == 'reg':
                acc_te = -1
            else:
                _, pred_l = torch.max(pred_cls, dim=1)
                target_l = te_cls.data
                acc_te += (torch.mean((pred_l == target_l).float())) * 100

                for true_label, predicted_label in zip(target_l, pred_l):
                    confusion_matrix[true_label, predicted_label] += 1
                print(confusion_matrix)

            # Loss
            if option['type_name'] == 'reg':
                loss_ = loss_te.data
            else:
                loss_ = -1

            print('Epoch : %d, Test Acc : %2.2f, Test Loss : %.2f,' % (epoch, acc_te, loss_))
            return model, loss_, acc_te