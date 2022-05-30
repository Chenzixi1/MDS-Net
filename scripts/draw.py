# -----------------------------------------
# python modules
# -----------------------------------------
from easydict import EasyDict as edict
from getopt import getopt
import numpy as np
import sys
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.core import *
from lib.imdb_util import *
from lib.loss.rpn_3d import *
from models.resnet_dilate import build

from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2


def draw_CAM(model, img_path, dep_path, save_path='view/out.png', resizeh=512, resizew=1760, isSave=False,
             isShow=False):
    # 图像加载&预处理
    rgb = Image.open(img_path).convert('RGB')
    loader = transforms.Compose([transforms.Resize(size=(resizeh, resizew)), transforms.ToTensor()])
    rgb = loader(rgb).unsqueeze(0)  # unsqueeze(0)在第0维增加一个维度
    # dep = Image.open(dep_path).convert('RGB')
    # loader = transforms.Compose([transforms.Resize(size=(resizeh, resizew)), transforms.ToTensor()])
    # dep = loader(dep).unsqueeze(0)  # unsqueeze(0)在第0维增加一个维度

    dep = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED)
    dep = cv2.resize(dep, (resizew, resizeh)).astype(np.float32)
    dep = dep - 4413.160626995486
    dep = dep / 3270.0158918863494
    dep = torch.from_numpy(dep).unsqueeze(0).unsqueeze(0)
    dep = torch.cat([dep, dep, dep], dim=1)

    # 获取模型输出的feature/score
    model.eval()  # 测试模式，不启用BatchNormalization和Dropout
    # feature = model.features(img, dep)
    # output = model.classifier(feature.view(1, -1))
    cls, prob, bbox_2d, bbox_3d, feat_size, rois, featureshows = model(rgb, dep)

    for j in range(len(featureshows)):
        feature = featureshows[j]

        output = feature.view(1, -1)

        print(cls.shape, feature.shape, output.shape)

        # 预测得分最高的那一类对应的输出score
        pred = torch.argmax(output).item()
        pred_class = output[:, pred]

        # 记录梯度值
        def hook_grad(grad):
            global feature_grad
            feature_grad = grad

        feature.register_hook(hook_grad)
        # 计算梯度
        pred_class.backward()

        grads = feature_grad  # 获取梯度

        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads,
                                                               (1, 1))  # adaptive_avg_pool2d自适应平均池化函数,输出大小都为（1，1）

        # 此处batch size默认为1，所以去掉了第0维（batch size维）
        pooled_grads = pooled_grads[0]  # shape为[batch,通道,size,size],此处batch为1，所以直接取[0]即取第一个batch的元素，就取到了每个batch内的所有元素
        features = feature[0]  # 取【0】原因同上

        ########################## 导数（权重）乘以相应元素
        # 512是最后一层feature的通道数
        for i in range(len(features)):
            features[i, ...] *= pooled_grads[i, ...]  # features[i, ...]与features[i]效果好像是一样的，都是第i个元素
        ##########################

        # 绘制热力图
        heatmap = features.detach().numpy()
        heatmap = np.mean(heatmap, axis=0)  # axis=0,对各列求均值，返回1*n

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        # 可视化原始热力图

        plt.matshow(heatmap)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.savefig('view/heatmap_' + str(j) + '.png')
        if isShow:
            plt.show()

        img = cv2.imread(img_path)  # 用cv2加载原始图像
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式

        # cv2.imwrite('view/heatmap.png', heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.5 + img  # 这里的0.4是热力图强度因子
        # 将图像保存到硬盘
        if isSave:
            cv2.imwrite('view/out_' + str(j) + '.png', superimposed_img)
            # 展示图像
        if isShow:
            superimposed_img /= 255
            plt.imshow(superimposed_img)

        cls, prob, bbox_2d, bbox_3d, feat_size, rois, featureshows = model(rgb, dep)


def main(argv):
    # -----------------------------------------
    # parse arguments
    # -----------------------------------------
    opts, args = getopt(argv, '', ['config=', 'restore='])

    # defaults
    conf_name = None
    restore = 135000

    # read opts
    for opt, arg in opts:

        if opt in ('--config'): conf_name = arg
        if opt in ('--restore'): restore = int(arg)

    # required opt
    if conf_name is None:
        raise ValueError('Please provide a configuration file name, e.g., --config=<config_name>')

    # -----------------------------------------
    # basic setup
    # -----------------------------------------

    # conf = init_config(conf_name)
    with open(
            'output/yolof_config/best_RowCatCol+SEdual+yolof_best_resnet_dilate50_batch2_dropoutearly0_5_lr0_0025_onecycle_iter160000_2021_07_20_23_33_32/conf.pkl',
            'rb') as file:
        conf = pickle.load(file)
        conf.use_sne = False

    paths = init_training_paths(conf_name, conf.result_dir)

    # init_torch(conf.rng_seed, conf.cuda_seed)
    init_log_file(paths.logs)

    # dataset = Dataset(conf, paths.data, paths.output)
    #
    # generate_anchors(conf, dataset.imdb, paths.output)
    # compute_bbox_stats(conf, dataset.imdb, paths.output)

    paths.output = os.path.join(paths.output, conf.result_dir)

    # -----------------------------------------
    # network and loss
    # -----------------------------------------

    # training network
    # rpn_net, _, scheduler = init_training_model(conf, paths.output, conf_name)

    rpn_net = build(conf, 'train')

    # resume training
    if restore:
        paths.weights = 'output/yolof_config/best_RowCatCol+SEdual+yolof_best_resnet_dilate50_batch2_dropoutearly0_5_lr0_0025_onecycle_iter160000_2021_07_20_23_33_32/weights/'
        modelpath = os.path.join(paths.weights, 'model_{}_pkl'.format(restore))

        pre_dict = torch.load(modelpath, map_location=torch.device('cpu'))
        new_pre = {}
        for k, v in pre_dict.items():
            name = k[7:]
            new_pre[name] = v
        rpn_net.load_state_dict(new_pre)
        # rpn_net.load_state_dict(torch.load(modelpath))

    return rpn_net


def show33(weight, name, show=False):
    name = 'view/' + name + '.png'

    weight = weight.permute(2, 3, 0, 1)
    weight = torch.mean(torch.mean(weight, dim=2), dim=2)
    a = weight.detach().numpy()
    min = np.amin(a)
    max = np.amax(a)
    a = (a - min) / (max - min)

    print(a)
    plt.matshow(a)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    # plt.gcf().set_size_inches(300 / 100, 300 / 100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(name)
    if show:
        plt.show()

def showrowcol(weight, name, col=1, show=False):
    # col = 1
    # row = 0
    name = 'view/' + name + '.png'

    weight = weight.permute(2, 3, 0, 1)
    weight = torch.mean(torch.mean(weight, dim=2), dim=2)
    weight = torch.cat([torch.zeros_like(weight), weight, torch.zeros_like(weight)], dim=col)
    a = weight.detach().numpy()

    min = np.amin(a)
    max = np.amax(a)
    a = (a - min) / (max - min)
    print(a)
    plt.matshow(a)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    # plt.gcf().set_size_inches(300 / 100, 300 / 100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(name)
    if show:
        plt.show()

    return a

def showtoge(weight, name, show=False):
    # col = 1
    # row = 0
    name = 'view/' + name + '.png'

    weight = weight
    min = np.amin(weight)
    max = np.amax(weight)
    weight = (weight - min) / (max - min)

    print(weight)

    plt.matshow(weight)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    # plt.gcf().set_size_inches(300 / 100, 300 / 100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(name)
    if show:
        plt.show()



def showweight(rpn_net):

    # show33(rpn_net.base.layer1[0].conv2.weight, name='layer1_depth_col_cmp')
    # layer1_depth_col = showrowcol(rpn_net.se_layer1.conv_depth_col.weight, name='layer1_depth_col', col=1)
    # layer1_rgb_col = showrowcol(rpn_net.se_layer1.conv_rgb_col.weight, name='layer1_rgb_col', col=1)
    # layer1_depth_row = showrowcol(rpn_net.se_layer1.conv_depth_row.weight, name='layer1_depth_row', col=0)
    # layer1_rgb_row = showrowcol(rpn_net.se_layer1.conv_rgb_row.weight, name='layer1_rgb_row', col=0)
    # showtoge(layer1_depth_col + layer1_depth_row, 'layer1_depth')
    # showtoge(layer1_rgb_col + layer1_rgb_row, 'layer1_rgb')
    #
    # show33(rpn_net.base.layer2[0].conv2.weight, name='layer2_depth_col_cmp')
    # layer2_depth_col = showrowcol(rpn_net.se_layer2.conv_depth_col.weight, name='layer2_depth_col', col=1)
    # layer2_rgb_col = showrowcol(rpn_net.se_layer2.conv_rgb_col.weight, name='layer2_rgb_col', col=1)
    # layer2_depth_row = showrowcol(rpn_net.se_layer2.conv_depth_row.weight, name='layer2_depth_row', col=0)
    # layer2_rgb_row = showrowcol(rpn_net.se_layer2.conv_rgb_row.weight, name='layer2_rgb_row', col=0)
    # showtoge(layer2_depth_col + layer2_depth_row, 'layer2_depth')
    # showtoge(layer2_rgb_col + layer2_rgb_row, 'layer2_rgb')
    #
    # show33(rpn_net.base.layer3[0].conv2.weight, name='layer3_depth_col_cmp')
    # layer3_depth_col = showrowcol(rpn_net.se_layer3.conv_depth_col.weight, name='layer3_depth_col', col=1)
    # layer3_rgb_col = showrowcol(rpn_net.se_layer3.conv_rgb_col.weight, name='layer3_rgb_col', col=1)
    # layer3_depth_row = showrowcol(rpn_net.se_layer3.conv_depth_row.weight, name='layer3_depth_row', col=0)
    # layer3_rgb_row = showrowcol(rpn_net.se_layer3.conv_rgb_row.weight, name='layer3_rgb_row', col=0)
    # showtoge(layer3_depth_col + layer3_depth_row, 'layer3_depth')
    # showtoge(layer3_rgb_col + layer3_rgb_row, 'layer3_rgb')

    show33(rpn_net.base.layer4[0].conv2.weight, name='layer4_depth_col_cmp')
    layer4_depth_col = showrowcol(rpn_net.se_layer4.conv_depth_col.weight, name='layer4_depth_col', col=1)
    layer4_rgb_col = showrowcol(rpn_net.se_layer4.conv_rgb_col.weight, name='layer4_rgb_col', col=1)
    layer4_depth_row = showrowcol(rpn_net.se_layer4.conv_depth_row.weight, name='layer4_depth_row', col=0)
    layer4_rgb_row = showrowcol(rpn_net.se_layer4.conv_rgb_row.weight, name='layer4_rgb_row', col=0)
    showtoge(layer4_depth_col + layer4_depth_row, 'layer4_depth')
    showtoge(layer4_rgb_col + layer4_rgb_row, 'layer4_rgb')


# run from command line
if __name__ == "__main__":
    rpn_net = main(sys.argv[1:])

    # showweight(rpn_net=rpn_net)

    # a = np.array([[0.0015, 0.0433, 0.0054],
    #               [0.0253, 0.0511, 0.0114],
    #               [0.0204, 0.0156, 0.0044]])
    # a = torch.from_numpy(a).type(torch.float32)
    # a = a.numpy()
    #
    # plt.matshow(a)
    # plt.axis('off')
    # plt.xticks([])
    # plt.yticks([])
    # # plt.savefig('view/heatmap_' + str(j) + '.png')
    # plt.show()

    # -----------------------------------------
    # view hitmap
    # -----------------------------------------

    # img_path = 'data/kitti_split/training/image_2/' + '000008.png'
    # dep_path = 'data/kitti_split/training/depth_2/' + '000008.png'

    # img_path = 'data/kitti_split/training/image_2/000008.png'
    # dep_path = 'data/kitti_split/training/depth_2/000008.png'

    # img_path = 'data/kitti_split/training/image_2/000475.png'
    # dep_path = 'data/kitti_split/training/depth_2/000475.png'

    # img_path = 'data/kitti_split/training/image_2/000443.png'
    # dep_path = 'data/kitti_split/training/depth_2/000443.png'

    # img_path = 'data/kitti_split/training/image_2/000422.png'
    # dep_path = 'data/kitti_split/training/depth_2/000422.png'

    img_path = 'data/kitti_split/training/image_2/000385.png'
    dep_path = 'data/kitti_split/training/depth_2/000385.png'

    # img_path = 'data/kitti_split/training/image_2/000262.png'
    # dep_path = 'data/kitti_split/training/depth_2/000262.png'

    draw_CAM(rpn_net, img_path, dep_path, 'view/out.png', isSave=True, isShow=False)
