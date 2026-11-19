import setting
import os

os.environ["CUDA_VISIBLE_DEVICES"] = setting.setting['gpu']
from tqdm import tqdm
from time import time
import torch
from sklearn.metrics import confusion_matrix
import torch.utils.data as dataA
from framework import MyFrame
from loss import dice_bce_loss
from data import *
from networks.testmy_tiao import CoBR



dataset = setting.setting['dataset']
SHAPE = setting.setting['crop_size']
# dataroot = setting.setting['dataroot']
model = setting.setting['model']
NAME = setting.setting['exper_name']
BATCHSIZE_PER_CARD = setting.setting['batch_size_card']
SEED = setting.setting['seed']
total_epoch = setting.setting['epoch']
mode = setting.setting['mode']

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# 设置保存路径
SAVE_ROOT = '/mnt/nfs/h1/haoqing/work1/'
WEIGHTS_DIR = os.path.join(SAVE_ROOT, 'weights')
LOGS_DIR = os.path.join(SAVE_ROOT, 'logs')

# 创建目录
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)



if model =='CoBR':
    solver = MyFrame(CoBR, dice_bce_loss, 2e-4)

batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

if dataset == 'dg':
    train_dir = os.path.join('/home/haoqing/dataset/dg/', 'train/image/')
    dataset = DPGlobe_Dataset(train_dir, type='Train')
    test_dir = os.path.join('/home/haoqing/dataset/dg/', 'test/image/')
    dataset_t = DPGlobe_Dataset(test_dir, type='Val')

if dataset == 'ma':
    train_dir = os.path.join('/home/haoqing/dataset/ma/', 'train/image/')
    dataset = Mass_Dataset(train_dir, type='Train')
    test_dir = os.path.join('/home/haoqing/dataset/ma/', 'test/image/')
    dataset_t = Mass_Dataset(test_dir, type='Val')

if dataset == 'cug':
    train_dir = os.path.join('/home/haoqing/dataset/cug/', 'train/image/')
    dataset = CUG(train_dir, type='Train')
    test_dir = os.path.join('/home/haoqing/dataset/cug/', 'test/image/')
    dataset_t = CUG(test_dir, type='Val')

if dataset == 'sp':
    train_dir = os.path.join('/home/haoqing/dataset/sp/', 'train/image/')
    dataset = SP(train_dir, type='Train')
    test_dir = os.path.join('/home/haoqing/dataset/sp/', 'test/image/')
    dataset_t = SP(test_dir, type='Val')

# if os.path.exists('weights/' + NAME + '.th'):# checkpoint
#     solver.load('weights/' + NAME + '.th')

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=4,
    drop_last=True  # 丢弃最后一个不完整的批次
)

data_loader_t = torch.utils.data.DataLoader(
    dataset_t,
    batch_size=batchsize,
    shuffle=False,
    num_workers=4,
    drop_last=True  # 丢弃最后一个不完整的批次
)

if not os.path.exists('logs/'):
    os.makedirs('logs/')

mylog = open(os.path.join(LOGS_DIR, NAME + '.log'), 'a')

tic = time()
total_train_time = 0
no_optim = 0
train_epoch_best_loss = 100.
best_iou = 0.0


# IoU 计算方法
class SegmentationMetric():
    def __init__(self, numclass):
        self.numclass = numclass
        self.confusionMatrix = np.zeros((self.numclass,) * 2)

    def CM(self, mask, pred):
        mask_num = mask.numpy().astype('int')
        pred = pred.astype('int')
        pred = pred.flatten()
        mask = mask_num.flatten()
        # 显式传递所有可能的标签（0和1），即使某些标签在当前批次中不存在
        cm = confusion_matrix(mask, pred, labels=[0, 1])
        self.confusionMatrix += cm

    def IOU(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        iou = intersection / union
        return iou


metric = SegmentationMetric(2)

for epoch in range(1, total_epoch + 1):  # checkpoint

    solver.net.train()

    train_tic = time()  # 记录当前epoch训练的起始时间
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in tqdm(data_loader_iter):
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss.item()
    train_epoch_loss /= len(data_loader_iter)

    train_toc = time()  # 记录当前epoch训练的结束时间
    train_time = train_toc - train_tic
    total_train_time += train_time

    print('********', file=mylog)
    print('epoch:', epoch, '    time:', int(total_train_time), file=mylog)
    print('train_loss:', train_epoch_loss, file=mylog)
    print('SHAPE:', SHAPE, file=mylog)
    print('********')
    print('epoch:', epoch, '    time:', int(total_train_time))
    print('train_loss:', train_epoch_loss)
    print('SHAPE:', SHAPE)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save(os.path.join(WEIGHTS_DIR, NAME + '_best_loss.th'))
        # 打印保存最佳模型的 epoch 和训练损失到日志
        print(f"Best model saved at epoch {epoch} with train_loss: {train_epoch_loss}", file=mylog)
        print(f"Best model saved at epoch {epoch} with train_loss: {train_epoch_loss}")

    # 每个epoch大于50后计算测试集IoU
    if epoch >= 50:
        # 评估模型在测试集上的表现
        solver.net.eval()
        metric.confusionMatrix = np.zeros((2, 2))  # 重置混淆矩阵
        with torch.no_grad():
            iou_total = 0
            for data_loader_iter1 in tqdm(iter(data_loader_t)):
                img, mask = data_loader_iter1
                img = img.cuda()
                pred = solver.net(img)
                pred[pred > 0.5] = 1
                pred[pred <= 0.5] = 0
                pred = pred.squeeze().cpu().data.numpy()
                metric.CM(mask, pred)

            iou = metric.IOU()[1]
            print(f'Epoch {epoch}:  IoU: {iou}', file=mylog)
            print(f'Epoch {epoch}:  IoU: {iou}')
            if iou > best_iou:
                best_iou = iou
                solver.save(os.path.join(WEIGHTS_DIR, NAME + '_best_iou.th'))
                # 创建一个长度为 100 的星号分隔符
                separator = "*" * 100
                print(separator, file=mylog)
                print(f"Best model saved with IoU: {iou}", file=mylog)
                print(f"Best model saved with IoU: {iou}")

    if epoch == int(total_epoch / 2):
        # solver.load('weights/'+NAME+'.th')
        # solver.save('weights/'+NAME+str(epoch)+'.th')
        solver.update_lr(5.0, factor=True, mylog=mylog)

    if epoch == int(total_epoch * 0.65):
        # solver.save('weights/'+NAME+str(epoch)+'.th')
        # solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor=True, mylog=mylog)

    if epoch == int(total_epoch * 0.8):
        # solver.save('weights/'+NAME+str(epoch)+'.th')
        # solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor=True, mylog=mylog)
    # if epoch == int(90):
    #     # solver.load('weights/'+NAME+'.th')
    #     # solver.save('weights/'+NAME+str(epoch)+'.th')
    #     solver.update_lr(5.0, factor=True, mylog=mylog)
    #
    # if epoch == int(110):
    #     # solver.save('weights/'+NAME+str(epoch)+'.th')
    #     # solver.load('weights/'+NAME+'.th')
    #     solver.update_lr(5.0, factor=True, mylog=mylog)
    #
    # if epoch == int(130):
    #     # solver.save('weights/'+NAME+str(epoch)+'.th')
    #     # solver.load('weights/'+NAME+'.th')
    #     solver.update_lr(5.0, factor=True, mylog=mylog)

    mylog.flush()

print('Finish!', file=mylog)
print('Finish!')
mylog.close()
