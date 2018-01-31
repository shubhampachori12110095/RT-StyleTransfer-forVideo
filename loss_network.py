import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import vgg


def vgg19(vgg_path, pretrained=True, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = vgg.VGG(vgg.make_layers(vgg.cfg['E']), **kwargs)
    if pretrained:
        state_dict = torch.load(vgg_path)
        state_dict = {k:v for k, v in state_dict.items() if 'class' not in k}
        model.load_state_dict(state_dict)
    return model


class LossNet(nn.Module):
    def __init__(self, vgg_path):
        model_list = list(vgg19(vgg_path).featurees.children())
        self.conv1_1 = model_list[0]
        self.conv1_2 = model_list[2]
        self.conv2_1 = model_list[5]
        self.conv2_2 = model_list[7]
        self.conv3_1 = model_list[10]
        self.conv3_2 = model_list[12]
        self.conv3_3 = model_list[14]
        self.conv3_4 = model_list[16]
        self.conv4_1 = model_list[19]
        self.conv4_2 = model_list[21]
        self.conv4_3 = model_list[23]
        self.conv4_4 = model_list[25]
        self.conv5_1 = model_list[28]
        self.conv5_2 = model_list[30]
        self.conv5_3 = model_list[32]
        self.conv5_4 = model_list[34]

    def forward(self, x, out_key):
        out = {}
        out['conv1_1'] = F.relu(self.conv1_1(x))
        out['conv1_2'] = F.relu(self.conv1_2(out['conv1_1']))
        out['pool1']   = F.max_pool2d(out['conv1_2'])

        out['conv2_1'] = F.relu(self.conv2_1(out['pool1']))
        out['conv2_2'] = F.relu(self.conv2_2(out['conv2_1']))
        out['pool2']   = F.max_pool2d(out['conv2_2'])
        
        out['conv3_1'] = F.relu(self.conv3_1(out['pool2']))
        out['conv3_2'] = F.relu(self.conv3_2(out['conv3_1']))
        out['conv3_3'] = F.relu(self.conv3_3(out['conv3_2']))
        out['conv3_4'] = F.relu(self.conv3_4(out['conv3_3']))
        out['pool3']   = F.max_pool2d(out['conv3_4'])

        out['conv4_1'] = F.relu(self.conv4_1(out['pool3']))
        out['conv4_2'] = F.relu(self.conv4_2(out['conv4_1']))
        out['conv4_3'] = F.relu(self.conv4_3(out['conv4_2']))
        out['conv4_4'] = F.relu(self.conv4_4(out['conv4_3']))
        out['pool4']   = F.max_pool2d(out['conv4_4'])

        out['conv5_1'] = F.relu(self.conv5_1(out['pool4']))
        out['conv5_2'] = F.relu(self.conv5_2(out['conv5_1']))
        out['conv5_3'] = F.relu(self.conv5_3(out['conv5_2']))
        out['conv5_4'] = F.relu(self.conv5_4(out['conv5_3']))
        out['pool5']   = F.max_pool2d(out['conv5_4'])

        return [out[key] for key in out_key]
