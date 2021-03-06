import torch
import torch.nn.functional as F
from module.model import AlexNet1D
import numpy as np
import pickle
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torchvision.transforms.transforms as transforms
import argparse

class GradCAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """

    def __init__(self, model):
        self.model_arch = model

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = torch.clamp(grad_output[0], min=0)
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        target_layer = self.model_arch.features._modules['12']
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, w = input[0].size()

        logit = self.model_arch(input[0], input[1])

        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = torch.unsqueeze(F.relu(saliency_map), dim=2)
        saliency_map = nn.Upsample(size=(1,w), mode='bilinear', align_corners=False)(saliency_map)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        saliency_map = saliency_map.squeeze()

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


class GradCAMpp(GradCAM):
    """Calculate GradCAM++ salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcampp
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcampp = GradCAMpp(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcampp(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """

    def __init__(self, model, verbose=False):
        super(GradCAMpp, self).__init__(model)

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, w = input[0].size()
        logit = self.model_arch(input[0], input[1])
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']  # dS/dA
        activations = self.activations['value']  # A
        b, k, u = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                      activations.mul(gradients.pow(3)).view(b, k, -1).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom + 1e-7)
        positive_gradients = F.relu(score.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha * positive_gradients).view(b, k, -1).sum(-1).view(b, k, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = torch.unsqueeze(F.relu(saliency_map), dim=2)
        saliency_map = nn.Upsample(size=(1,w), mode='bilinear', align_corners=False)(saliency_map)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        saliency_map = saliency_map.squeeze()
        return saliency_map, logit

## ArgParse
arg = argparse.ArgumentParser()
arg.add_argument('--exp_num', default=0, type=int)
arg.add_argument('--fusion', default='mid', type=str)
args = arg.parse_args()

# Option
gpu = True
pp = True

exp_num = args.exp_num
pt_name = '../result/0223_final_late/%d/best_model.pt' %exp_num
fusion = 'mid'
save_folder = '../grad_result/%s/%d' %(fusion, exp_num)

# Reference
with open('../Data_pre/MeatData_0213.pkl', 'rb') as f:
    data_meat = pickle.load(f)

spec = data_meat[0][2][30:-1487]

# Original Dataset
with open('../met_fusion/dataset_cut.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Train and Test dataset
tr_x = dataset['tr_x']
tr_y = dataset['tr_y']
tr_aux = dataset['tr_aux']

tr_m, tr_s = np.mean(tr_x), np.std(tr_x)

te_x = dataset['te_x']
te_y = dataset['te_y']
te_aux = dataset['te_aux']

# To tensor
transform_x = transforms.Normalize([tr_m], [tr_s])

tr_ori_x = torch.unsqueeze(torch.from_numpy(tr_x).float(), dim=1)
tr_x = transform_x(tr_ori_x)

tr_aux = torch.unsqueeze(torch.from_numpy(tr_aux).float(), dim=2)
tr_y = torch.from_numpy(tr_y).long()

te_ori_x = torch.unsqueeze(torch.from_numpy(te_x).float(), dim=1)
te_x = transform_x(te_ori_x)

te_aux = torch.unsqueeze(torch.from_numpy(te_aux).float(), dim=2)
te_y = torch.from_numpy(te_y).long()

# Considering the fusion
option = {}

# Model Definition
out_channel = 512
model_f = AlexNet1D

model = model_f(in_channel=1, train_rule=fusion)

if gpu == True:
    model = model.cuda()

# Load Model
saved = torch.load(pt_name)['model']
model.load_state_dict(saved)

mask_dict = {'0':[], '1':[], '2':[]}
graph_dict = {'0':[], '1':[], '2':[]}

print('Grad CAM ++ for training dataset')
for ix in range(int(tr_x.size()[0])):
    sample_x = tr_x[[ix]].cuda()
    sample_aux = tr_aux[[ix]].cuda().view(1,1,-1)
    sample_y = tr_y[[ix]].cuda()

    # GradCAM
    gradcam = GradCAM(model)
    mask, _ = gradcam([sample_x, sample_aux], class_idx=int(sample_y))

    gradcam_plus = GradCAMpp(model)
    mask_plus, _ = gradcam_plus([sample_x, sample_aux], class_idx=int(sample_y))

    # Visualize
    if pp == True:
        del mask
        mask = mask_plus.cpu().numpy()
    else:
        del mask_plus
        mask = mask.cpu().numpy()

    graph = tr_ori_x[[ix]].squeeze().numpy().reshape(1,-1)

    cls_ix = int(sample_y)
    mask_dict[str(cls_ix)].append(mask.reshape(-1))
    graph_dict[str(cls_ix)].append(graph)

## Figure for train
spec = np.asarray(spec).reshape(-1)
os.makedirs(save_folder, exist_ok=True)
for cls_ix in ['0','1','2']:
    tr_mask = np.mean(np.asarray(mask_dict[cls_ix]), axis=0).reshape(-1)
    tr_graph = np.mean(np.asarray(graph_dict[cls_ix]), axis=0).reshape(-1)

    tr_mask = tr_mask[np.newaxis,:]

    # Save the result as excel
    exc = pd.DataFrame(np.concatenate([spec.reshape(-1,1), tr_graph.reshape(-1,1), tr_mask.reshape(-1,1)], axis=1), columns=['Wavelength','Reflectance','GradWeight'])
    exc.to_csv(os.path.join(save_folder,'%s_pp_%s_tr_class%s.csv' %(fusion, pp, cls_ix)), index=False)

    fig = plt.figure()
    plt.imshow(tr_mask, cmap='coolwarm', aspect='auto', extent=[min(spec), max(spec),0,1.2])
    plt.colorbar()
    plt.plot(spec, tr_graph)

    plt.ylim((0,1.2))
    plt.title('GradCAM-Average:class %s' %cls_ix)
    plt.xlabel('Wavelength')
    plt.ylabel('Reflectance')
    plt.savefig(os.path.join(save_folder,'%s_pp_%s_tr_class%s.png' %(fusion, pp, cls_ix)))
    plt.close()

# For test dataset
print('Grad CAM ++ for evaluation dataset')
del mask_dict, graph_dict, sample_x, sample_aux, sample_y, tr_mask, tr_graph

mask_dict = {'0':[], '1':[], '2':[]}
graph_dict = {'0':[], '1':[], '2':[]}

for ix in range(int(te_x.size()[0])):
    sample_x = te_x[[ix]].cuda()
    sample_aux = te_aux[[ix]].cuda().view(1,1,-1)
    sample_y = te_y[[ix]].cuda()

    # GradCAM
    gradcam = GradCAM(model)
    mask, _ = gradcam([sample_x, sample_aux], class_idx=int(sample_y))

    gradcam_plus = GradCAMpp(model)
    mask_plus, _ = gradcam_plus([sample_x,sample_aux], class_idx=int(sample_y))

    # Visualize
    if pp == True:
        del mask
        mask = mask_plus.cpu().numpy()
    else:
        del mask_plus
        mask = mask.cpu().numpy()

    graph = te_ori_x[[ix]].squeeze().numpy().reshape(1,-1)

    cls_ix = int(sample_y)
    mask_dict[str(cls_ix)].append(mask.reshape(-1))
    graph_dict[str(cls_ix)].append(graph)

for cls_ix in ['0','1','2']:
    te_mask = np.mean(np.asarray(mask_dict[cls_ix]), axis=0).reshape(-1)
    te_graph = np.mean(np.asarray(graph_dict[cls_ix]), axis=0).reshape(-1)

    te_mask = te_mask[np.newaxis,:]

    # Save the result as excel
    exc = pd.DataFrame(np.concatenate([spec.reshape(-1,1), te_graph.reshape(-1,1), te_mask.reshape(-1,1)], axis=1), columns=['Wavelength','Reflectance','GradWeight'])
    exc.to_csv(os.path.join(save_folder,'%s_pp_%s_te_class%s.csv' %(fusion, pp, cls_ix)), index=False)

    fig = plt.figure()
    plt.imshow(te_mask, cmap='coolwarm', aspect='auto', extent=[min(spec), max(spec),0,1.2])
    plt.colorbar()
    plt.plot(spec, te_graph)

    plt.ylim((0.0,1.2))
    plt.title('GradCAM-Average:class %s' %cls_ix)
    plt.xlabel('Wavelength')
    plt.ylabel('Reflectance')
    plt.savefig(os.path.join(save_folder,'%s_pp_%s_te_class%s.png' %(fusion, pp, cls_ix)))
    plt.close()