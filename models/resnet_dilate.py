import torch.nn as nn
from lib.rpn_util import *
from models import resnet
import torch
import numpy as np
from models.se_module import SqueezeAndExcitation
from models.encoder import DilatedEncoder

class CrossMixMoudle(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(CrossMixMoudle, self).__init__()

        self.channels_out = int(channels_in / 2)

        self.se_rgb = SqueezeAndExcitation(channels_in, activation=activation)

        self.conv_rgb_row = nn.Conv2d(channels_in, self.channels_out, (1, 3), padding=(0, 1))
        self.conv_rgb_col = nn.Conv2d(channels_in, self.channels_out, (3, 1), padding=(1, 0))
        self.conv_depth_row = nn.Conv2d(channels_in, self.channels_out, (1, 3), padding=(0, 1))
        self.conv_depth_col = nn.Conv2d(channels_in, self.channels_out, (3, 1), padding=(1, 0))

        self.distribution = nn.Conv2d(channels_in, channels_in, (1, 1), padding=0)
        self.bn = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU(inplace=True)

        self.se = SqueezeAndExcitation(channels_in, activation=activation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, rgb, depth, show=False):

        depth0 = depth
        rgb0 = rgb

        rgb_row = self.conv_rgb_row(rgb)
        rgb_col = self.conv_rgb_col(rgb)
        depth_row = self.conv_depth_row(depth)
        depth_col = self.conv_depth_col(depth)

        merge = torch.cat([rgb_row + rgb_col, depth_row + depth_col], dim=1)
        merge = self.distribution(merge)
        merge = self.bn(merge)
        merge = self.relu(merge)
        merge0 = merge

        merge = self.se(merge)

        rgb = self.se_rgb(rgb)

        out = rgb + merge

        if show:
            return out, depth0, rgb0, merge0, rgb, merge
        else:
            return out, depth0


class RPN(nn.Module):

    def __init__(self, phase, conf):
        super(RPN, self).__init__()

        self.base = resnet.ResNetDilate(conf.base_model)
        self.adaptive_diated = conf.adaptive_diated
        self.dropout_position = conf.dropout_position
        self.use_dropout = conf.use_dropout
        self.drop_channel = conf.drop_channel
        self.use_corner = conf.use_corner
        self.corner_in_3d = conf.corner_in_3d
        self.deformable = conf.deformable

        self.depthnet = resnet.ResNetDilate(50)

        # settings
        self.phase = phase
        self.num_classes = len(conf['lbls']) + 1
        self.num_anchors = conf['anchors'].shape[0]

        # self.prop_feats = nn.Sequential(
        #     nn.Conv2d(2048, 512, 3, padding=1),
        #     nn.ReLU(inplace=True),
        # )
        self.prop_feats = DilatedEncoder(2048)
        self.out_channel = self.prop_feats.out_channels

        if self.use_dropout:
            self.dropout = nn.Dropout(p=conf.dropout_rate)

        if self.drop_channel:
            self.dropout_channel = nn.Dropout2d(p=0.3)

        # outputs
        self.cls = nn.Conv2d(self.out_channel, self.num_classes * self.num_anchors, 1)

        # bbox 2d
        self.bbox_x = nn.Conv2d(self.out_channel, self.num_anchors, 1)
        self.bbox_y = nn.Conv2d(self.out_channel, self.num_anchors, 1)
        self.bbox_w = nn.Conv2d(self.out_channel, self.num_anchors, 1)
        self.bbox_h = nn.Conv2d(self.out_channel, self.num_anchors, 1)

        # bbox 3d
        self.bbox_x3d = nn.Conv2d(self.out_channel, self.num_anchors, 1)
        self.bbox_y3d = nn.Conv2d(self.out_channel, self.num_anchors, 1)
        self.bbox_z3d = nn.Conv2d(self.out_channel, self.num_anchors, 1)
        self.bbox_w3d = nn.Conv2d(self.out_channel, self.num_anchors, 1)
        self.bbox_h3d = nn.Conv2d(self.out_channel, self.num_anchors, 1)
        self.bbox_l3d = nn.Conv2d(self.out_channel, self.num_anchors, 1)
        self.bbox_rY3d = nn.Conv2d(self.out_channel, self.num_anchors, 1)

        if self.corner_in_3d:
            self.bbox_3d_corners = nn.Conv2d(self.out_channel, self.num_anchors * 18, 1)  # 2 * 8 + 2
            self.bbox_vertices = nn.Conv2d(self.out_channel, self.num_anchors * 24, 1)  # 3 * 8
        elif self.use_corner:
            self.bbox_vertices = nn.Conv2d(self.out_channel, self.num_anchors * 24, 1)


        self.softmax = nn.Softmax(dim=1)

        self.feat_stride = conf.feat_stride
        self.feat_size = calc_output_size(np.array(conf.crop_size), self.feat_stride)
        self.rois = locate_anchors(conf.anchors, self.feat_size, conf.feat_stride, convert_tensor=True)
        self.rois = self.rois.type(torch.cuda.FloatTensor)
        self.anchors = conf.anchors

        # [1, 64, 256, 880]
        # [1, 256, 128, 440]
        # [1, 512, 64, 220]
        # [1, 1024, 32, 110]
        # [1, 2048, 32, 110]
        self.se_layer1 = CrossMixMoudle(256, activation=nn.ReLU(inplace=True))
        self.se_layer2 = CrossMixMoudle(512, activation=nn.ReLU(inplace=True))
        self.se_layer3 = CrossMixMoudle(1024, activation=nn.ReLU(inplace=True))
        self.se_layer4 = CrossMixMoudle(2048, activation=nn.ReLU(inplace=True))

        # diliate encoder


    def forward(self, x, depth):

        features = []

        batch_size = x.size(0)

        # ESANet function
        x = self.base.conv1(x)
        depth = self.depthnet.conv1(depth)
        x = self.base.bn1(x)
        depth = self.depthnet.bn1(depth)
        x = self.base.relu(x)
        depth = self.depthnet.relu(depth)
        x = self.base.maxpool(x)
        depth = self.depthnet.maxpool(depth)

        x = self.base.layer1(x)
        depth = self.depthnet.layer1(depth)
        x, depth = self.se_layer1(x, depth)

        # x, depth, rgb0, merge0, rgb, merge = self.se_layer1(x, depth. True)
        # features.append(rgb0)
        # features.append(rgb)
        # features.append(merge0)
        # features.append(merge)
        # features.append(x)
        # features.append(depth)


        # -------------------------------------------------------------------- #
        x = self.base.layer2(x)
        depth = self.depthnet.layer2(depth)
        x, depth = self.se_layer2(x, depth)

        # x, depth, rgb0, merge0, rgb, merge = self.se_layer2(x, depth, True)
        # features.append(rgb0)
        # features.append(rgb)
        # features.append(merge0)
        # features.append(merge)
        # features.append(x)
        # features.append(depth)

        # -------------------------------------------------------------------- #
        x = self.base.layer3(x)
        depth = self.depthnet.layer3(depth)
        x, depth = self.se_layer3(x, depth)

        # x, depth, rgb0, merge0, rgb, merge = self.se_layer3(x, depth, True)
        # features.append(rgb0)
        # features.append(rgb)
        # features.append(merge0)
        # features.append(merge)
        # features.append(x)
        # features.append(depth)

        # -------------------------------------------------------------------- #

        x = self.base.layer4(x)
        depth = self.depthnet.layer4(depth)
        x, depth = self.se_layer4(x, depth)

        # x, depth, rgb0, merge0, rgb, merge = self.se_layer4(x, depth, True)
        # features.append(rgb0)
        # features.append(rgb)
        # features.append(merge0)
        # features.append(merge)
        # features.append(x)
        # features.append(depth)

        x = x * depth

        # 3d detection head
        if self.use_dropout and self.dropout_position == 'early':
            x = self.dropout(x)

        prop_feats = self.prop_feats(x)

        if self.use_dropout and self.dropout_position == 'late':
            prop_feats = self.dropout(prop_feats)

        cls = self.cls(prop_feats)

        # bbox 2d
        bbox_x = self.bbox_x(prop_feats)
        bbox_y = self.bbox_y(prop_feats)
        bbox_w = self.bbox_w(prop_feats)
        bbox_h = self.bbox_h(prop_feats)

        # bbox 3d
        bbox_x3d = self.bbox_x3d(prop_feats)
        bbox_y3d = self.bbox_y3d(prop_feats)
        bbox_z3d = self.bbox_z3d(prop_feats)
        bbox_w3d = self.bbox_w3d(prop_feats)
        bbox_h3d = self.bbox_h3d(prop_feats)
        bbox_l3d = self.bbox_l3d(prop_feats)
        bbox_rY3d = self.bbox_rY3d(prop_feats)
        # targets_dx, targets_dy, delta_z, scale_w, scale_h, scale_l, deltaRotY


        feat_h = cls.size(2)
        feat_w = cls.size(3)

        # reshape for cross entropy
        cls = cls.view(batch_size, self.num_classes, feat_h * self.num_anchors, feat_w)

        # score probabilities
        prob = self.softmax(cls)

        # reshape for consistency
        # although it's the same with x.view(batch_size, -1, 1) when c == 1, useful when c > 1
        bbox_x = flatten_tensor(bbox_x.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y = flatten_tensor(bbox_y.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w = flatten_tensor(bbox_w.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h = flatten_tensor(bbox_h.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        bbox_x3d = flatten_tensor(bbox_x3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y3d = flatten_tensor(bbox_y3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_z3d = flatten_tensor(bbox_z3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w3d = flatten_tensor(bbox_w3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h3d = flatten_tensor(bbox_h3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_l3d = flatten_tensor(bbox_l3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_rY3d = flatten_tensor(bbox_rY3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        # bundle
        bbox_2d = torch.cat((bbox_x, bbox_y, bbox_w, bbox_h), dim=2)
        bbox_3d = torch.cat((bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_rY3d), dim=2)

        if self.corner_in_3d:
            corners_3d = self.bbox_3d_corners(prop_feats)
            corners_3d = flatten_tensor(corners_3d.view(batch_size, 18, feat_h * self.num_anchors, feat_w))
            bbox_vertices = self.bbox_vertices(prop_feats)
            bbox_vertices = flatten_tensor(bbox_vertices.view(batch_size, 24, feat_h * self.num_anchors, feat_w))
        elif self.use_corner:
            bbox_vertices = self.bbox_vertices(prop_feats)
            bbox_vertices = flatten_tensor(bbox_vertices.view(batch_size, 24, feat_h * self.num_anchors, feat_w))

        feat_size = [feat_h, feat_w]

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        if self.training:
            if self.corner_in_3d:
                return cls, prob, bbox_2d, bbox_3d, torch.from_numpy(
                    np.array(feat_size)).cuda(), bbox_vertices, corners_3d
            elif self.use_corner:
                return cls, prob, bbox_2d, bbox_3d, torch.from_numpy(np.array(feat_size)).cuda(), bbox_vertices
            else:
                return cls, prob, bbox_2d, bbox_3d, torch.from_numpy(np.array(feat_size)).cuda()

        else:

            if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
                self.feat_size = [feat_h, feat_w]
                self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride, convert_tensor=True)
                self.rois = self.rois.type(torch.cuda.FloatTensor)

            return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois, features


def build(conf, phase='train'):
    train = phase.lower() == 'train'

    rpn_net = RPN(phase, conf)
    print(rpn_net)
    if train:
        rpn_net.train()
    else:
        rpn_net.eval()

    return rpn_net
