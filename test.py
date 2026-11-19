import os
import setting
os.environ["CUDA_VISIBLE_DEVICES"]= setting.setting['gpu']
import torch.utils.data as data

import time
from tqdm import tqdm
# from thop import profile


from sklearn.metrics import confusion_matrix

from data import *
from networks.testmy_tiao import CoBR

NAME = setting.setting['exper_name']
BATCHSIZE_PER_CARD = setting.setting['batch_size_card']
SEED = setting.setting['seed']
out_pred = setting.setting['out_pred']
dataset = setting.setting['dataset']
model = setting.setting['model']
shape = setting.setting['crop_size']

target = './results/'+NAME+'_'+dataset+'/'
# 确保目标文件夹存在
if out_pred == True:
    if not os.path.exists(target):
        os.makedirs(target)

if out_pred == True:
    if not os.path.isdir(target):
        os.makedirs(target)


if dataset == 'dg':
    test_dir = os.path.join('/home/haoqing/dataset/dg/', 'test/image/')
    dataset = DPGlobe_Dataset(test_dir,type='Test')
if dataset == 'ma':
    test_dir = os.path.join('/home/haoqing/dataset/ma/', 'test/image/')
    dataset = Mass_Dataset(test_dir, type='Test')
if dataset == 'cug':
    test_dir = os.path.join('/home/haoqing/dataset/cug/', 'test/image/')
    dataset = CUG(test_dir, type='Test')



mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


if model == 'CoBR':
    net = CoBR()


net = net.cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

# 根据 mode 选择加载不同的模型
mode = setting.setting['mode']  # 假设 mode 是保存在 setting 中的

# 判断 mode 的值来加载不同的模型
if mode == 'loss':
    # 加载保存了最佳训练损失的模型
    model_path = './weights/' + NAME + '_best_loss.th'
else:
    # 默认加载最后保存的模型
    model_path = './weights/' + NAME + '.th'

# model_path =  "/home/haoqing/seg_road/OARENet-main/weights/SwinT_OAM_DP.th"

# 加载模型
net.load_state_dict(torch.load(model_path))

# net.load_state_dict(torch.load('./weights/'+NAME+'.th'))

batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=False,
    num_workers=4)

tic = time.time()
class SegmentationMetric():
    def __init__(self,numclass):
        self.numclass = numclass
        self.confusionMatrix = np.zeros((self.numclass,) * 2)

    def CM(self, mask, pred):
        mask_num = mask.numpy().astype('int')
        pred = pred.astype('int')
        pred = pred.flatten()
        mask = mask_num.flatten()
        cm = confusion_matrix(mask,pred)
        self.confusionMatrix += cm
        # return cm

    def PA(self):
        # PA = acc
        acc = np.diag(self.confusionMatrix).sum()/self.confusionMatrix.sum()
        return acc

    def CPA(self):
        # CPA = (tp)/tp+fp
        # precision
        cpa = np.diag(self.confusionMatrix)/self.confusionMatrix.sum(axis=0)
        return cpa

    def meanPA(self):
        cpa = self.CPA()
        meanpa = np.nanmean(cpa)
        return meanpa
    def CPR(self):
        # CPR:class recall
        cpr = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return cpr

    def IOU(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        iou = intersection/union
        return iou

metric = SegmentationMetric(2)
# cpa1=[]
# cpr1=[]
# f11=[]
# iou1=[]
title = 0
net.eval()

with torch.no_grad():
    for data_loader_iter in tqdm(iter(data_loader)):
        img, mask = data_loader_iter
        img = img.cuda()
        # solver.set_input(img, mask)
        # 设置 epoch 和 hw_range 参数（可以根据需要调整这些值）
        epoch = 150  # 可以根据实际情况设置
        hw_range = [0, 18]  # 设置适当的 hw_range

        # 修改为：传递 epoch 和 hw_range 参数
        pred = net(img, epoch, hw_range)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        if out_pred==True:
            for i in range(0,pred.shape[0]):
                maskout = mask[i].cpu()
                imgout = img[i].cpu()
                predout = pred[i].cpu()
                cv2.imwrite(target +str(title)+'_mask.png', (maskout.numpy().astype(np.uint8)*255).transpose(1,2,0))
                cv2.imwrite(target + str(title) + '_img.png', (((imgout.numpy()*std)+mean)*255).astype(np.uint8).transpose(1,2,0))
                cv2.imwrite(target + str(title) + '_pred.png', (predout.numpy().astype(np.uint8)*255).transpose(1,2,0))
                title += 1
        pred = pred.squeeze().cpu().data.numpy()
        # mask.shape(3,1,768,768) pred.shape(3,768,768)
        metric.CM(mask, pred)

cpa = metric.CPA()[1]
cpr = metric.CPR()[1]
f1 = 2*(cpa*cpr/(cpa+cpr))
iou = metric.IOU()[1]
# 写入实验结果
metric_file = open('./logs/' + NAME + '_metric.log', 'a')
# 写入实验条件
metric_file.write('--------%s--------\n' % str(time.strftime('%Y-%m-%d %H:%M',time.localtime(time.time()))))
for k,v in setting.setting.items():
    metric_file.write('{}:{}\n'.format(k, v))
metric_file.write('cpa:{}\n'.format(cpa))
metric_file.write('cpr:{}\n'.format(cpr))
metric_file.write('f1:{}\n'.format(f1))
metric_file.write('iou:{}\n'.format(iou))
print('Finish!')
metric_file.close()

# # 假设真实标签是 mask，模型的预测是 pred
# mask = np.array([1, 1, 1, 0, 1, 0, 1, 0])  # 示例真实标签
# pred = np.array([1, 1, 0, 0, 1, 0, 1, 1])  # 示例模型预测结果
#
# # 将 mask 和 pred 转换为 PyTorch 张量
# mask_tensor = torch.tensor(mask, dtype=torch.int)  # 转换为 int 类型张量
# pred_tensor = torch.tensor(pred, dtype=torch.int)  # 转换为 int 类型张量
#
# # 初始化评价指标类，numclass 为2（背景和前景）
# metric = SegmentationMetric(numclass=2)
#
# # 计算混淆矩阵
# cm = metric.CM(mask_tensor, pred)
# print("Confusion Matrix:")
# print(cm)
#
# # 计算像素精度
# accuracy = metric.PA()
# print("Pixel Accuracy:", accuracy)
#
# # 计算每个类别的精确度
# precision = metric.CPA()
# print("Class Precision:", precision)
#
# # 计算平均精度
# mean_precision = metric.meanPA()
# print("Mean Precision:", mean_precision)
#
# # 计算每个类别的召回率
# recall = metric.CPR()
# print("Class Recall:", recall)
#
# # 计算每个类别的IoU
# iou = metric.IOU()
# print("Class IoU:", iou)
