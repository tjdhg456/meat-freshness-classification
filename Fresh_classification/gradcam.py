import torch
import torch.nn.functional as F
from module.model import AlexNet1D, EMB_cls
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import os

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

        target_layer = self.model_arch.model.features._modules['11']
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
        b, c, w = input.size()

        logit = self.model_arch(input)

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
        saliency_map = F.upsample(saliency_map, size=(w,1), mode='bilinear', align_corners=False)
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
        b, c, w = input.size()
        logit = self.model_arch(input)
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
        saliency_map = F.upsample(saliency_map, size=(w,1), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        saliency_map = saliency_map.squeeze()
        return saliency_map, logit


## main
# Option
np.random.seed(42)
torch.manual_seed(10)

# Original Dataset
with open('./dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

train_dataset = dataset['train']
test_dataset = dataset['test']
spec = dataset['wavelength']

# To numpy array (Train/Test Split)
tr_x = train_dataset[:, :-1]
tr_y = train_dataset[:,-1]
te_x = test_dataset[:,:-1]
te_y = test_dataset[:,-1]

# To torch tensor
tr_x = torch.from_numpy(tr_x).float()
tr_y = torch.from_numpy(tr_y).long()
te_x = torch.from_numpy(te_x).float()
te_y = torch.from_numpy(te_y).long()

# sample
sample_x = torch.unsqueeze(tr_x, dim=1)[[0]].cuda()
sample_y = tr_y[0]

# Load Model
pt_name = './result3/model0_sampler3_91.07.pt'
saved = torch.load(pt_name)

model_base = AlexNet1D(num_classes=2, result_emb=True)
model_base.up_type('cls')
model_base = model_base.cuda()
model = EMB_cls(model=model_base, out_class=3)
model.load_state_dict(saved['model'])

gradcam = GradCAM(model)
mask, logit = gradcam(sample_x, class_idx=int(sample_y))

gradcam_plus = GradCAMpp(model)
mask_plus, logit_plus = gradcam_plus(sample_x, class_idx=int(sample_y))
print(mask_plus)

# Visualize
length = 10
y = np.linspace(0,1,num=length)
z = [[i for i in mask_plus] for j in y]

os.makedirs('./grad_result', exist_ok=True)

plt.figure()
plt.contourf(spec, y, z, alpha=0.6)
plt.plot(spec, sample_x.squeeze().cpu().numpy())
plt.ylim((0,1))
plt.title('GradCAM-Average')
plt.xlabel('Wavelength')
plt.savefig('./grad_result/ex2.png')
plt.close()
