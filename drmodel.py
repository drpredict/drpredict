from torchvision.models import ResNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import Bottleneck

from logs import get_logger
from torch.utils.data import Dataset
from augment import Compose, ToFloat, ToTensor
from model.maskrcnn import MaskRCNN
from torch.optim import SGD
from sqlitedict import SqliteDict
from io import BytesIO
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

logger = get_logger('main')

class TrainEvalDataset(Dataset):
    def __init__(self, data_reader, split='train'):
        super().__init__()
        self.data_reader = data_reader
        if split == 'train':
            transform = [
                ToFloat(),
                ToTensor(),
            ]
        else:
            transform = [
                ToFloat(),
                ToTensor(),
            ]
        self.transform = Compose(transform)

    def __getitem__(self, index):
        try:
            image, label = self.data_reader[index]
            image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(e, exc_info=True)

    def __len__(self):
        return self.data_reader.__len__()


class PredBackbone(ResNet):
    def __init__(self, **kwargs):
        super(PredBackbone, self).__init__(Bottleneck, [3, 4, 23, 3], **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        result = dict(feature=x)
        # x = self.fc(x)
        # result['logits'] = x
        return  result



class DeepDRModule(nn.Module):
    def __init__(
            self,
            snap_storage=None,
            load_from_epoch=-1,
            device='cuda',
            trainable_layers=5,
            lr=0.001,
            num_classes=300) -> None:
        super().__init__()
        self.storage_dict = None
        if snap_storage is not None:
            os.makedirs(os.path.split(snap_storage)[0], exist_ok=True)
            self.storage_dict = SqliteDict(snap_storage)
        self.maskrcnn = MaskRCNN(num_class=5, trainable_layers=5)

        self.features = resnet_fpn_backbone('resnet101', True, trainable_layers=trainable_layers)
        self.f1 = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # self.l1 = nn.Linear(num_classes, num_classes, bias=False)
        # self.theta_t2 = nn.Softmax(dim=1)

        self.device = torch.device(device)
        self.maskrcnn = self.maskrcnn.to(self.device)
        self.features.to(self.device)
        self.f1.to(self.device)
        # self.theta_t2.to(self.device)
        self.epoch=0
        self.lr = lr

        if self.storage_dict is not None and len(self.storage_dict) > 0:
            kk = list(self.storage_dict.keys())[-1]
            if load_from_epoch >= 0:
                kk = load_from_epoch
            self.load_state_dict(
                torch.load(BytesIO(self.storage_dict[kk])))
            start_epoch = int(kk) + 1
            logger.info(f'loading from epoch{start_epoch}')
            self.epoch = int(kk)

    def snap_shot(self):
        if self.storage_dict is None:
            return
        logger.debug(f'saving epoch {self.epoch}')
        buffer = BytesIO()
        torch.save(self.state_dict(), buffer)
        buffer.seek(0)
        self.storage_dict[self.epoch] = buffer.read()
        self.storage_dict.commit()

    def forward_maskrcnn(self, image, target=None):
        return self.maskrcnn(image, target)

    def train_mask_rcnn_epoch(self, loader):
        optimizer = SGD(self.maskrcnn.parameters(), self.lr, 0.95, weight_decay=0.00001)
        self.epoch += 1
        self.maskrcnn.train()
        process_bar = tqdm(enumerate(loader), total=len(loader))
        for batch_cnt, batch in process_bar:
            image, label = batch
            for k, v in label.items():
                label[k] = v.squeeze()
            image = image.to(self.device)
            for k, v in label.items():
                if isinstance(v, torch.Tensor):
                    label[k] = label[k].to(self.device)
            # print(label['boxes'].shape)
            optimizer.zero_grad()
            net_out = self.maskrcnn(image, [label])
            loss = 0
            for i in net_out.values():
                loss += i  
            net_out['loss_sum'] = loss
            loss.backward()
            optimizer.step()
            process_bar.set_description_str(f'loss: {float(loss):.3f}', True)
        # exp_lr_scheduler.step(epoch)
        self.snap_shot()

    def forward_classification(self, x, with_mask_rcnn=False):
        x2=x
        x = self.features(x)['pool']
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.size(0), -1)

        if with_mask_rcnn:
            x2 = self.maskrcnn.backbone(x2)['pool']
            x2 = torch.nn.functional.adaptive_avg_pool2d(x2, (1, 1)).view(x2.size(0), -1)
            x2 = x2.fill_(0)
        else:
            x2 = torch.zeros(x.size(0), 256, dtype=x.dtype, device=x.device)
        x = torch.cat((x2, x), dim=1)
        # x = self.dropout(x)
        t1 = self.f1(x)
        # y_hat = self.theta_t1(t1)
        return t1

    def train_classification(self, loader, with_mask_rcnn=False):
        self.features.train()
        self.f1.train()
        params = [
            {'params': self.features.parameters(), 'lr': self.lr},
            {'params': self.f1.parameters(), 'lr': self.lr}
        ]
        optimizer = SGD(params, self.lr, 0.9, weight_decay=0.00001)
        # exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, 0.95)
        self.epoch += 1
        if with_mask_rcnn:
            self.maskrcnn.train()
        process_bar = tqdm(enumerate(loader), total=len(loader))
        for batch_cnt, batch in process_bar:
            image, label = batch
            label = label.squeeze()
            image = image.to(self.device)
            label = label.to(self.device)
            optimizer.zero_grad()
            pred = self.forward_classification(image, with_mask_rcnn=with_mask_rcnn)
            loss = F.cross_entropy(pred, label)
            loss.backward()
            optimizer.step()
            process_bar.set_description_str(f'loss: {float(loss):.3f}', True)
        # exp_lr_scheduler.step(epoch)
        self.snap_shot()

    def dump_maskrcnn(self, path):
        if os.path.split(path)[0] != '':
            os.makedirs(os.path.split(path)[0], exist_ok=True)
        torch.save(self.maskrcnn.state_dict(), open(path, 'wb'))

    def dump_all(self, path):
        if os.path.split(path)[0] != '':
            os.makedirs(os.path.split(path)[0], exist_ok=True)
        torch.save(self.state_dict(), open(path, 'wb'))

    def dump_classification(self, path):
        if os.path.split(path)[0] != '':
            os.makedirs(os.path.split(path)[0], exist_ok=True)
        torch.save((self.features.state_dict(), self.f1.state_dict()),
                   open(path, 'wb'))

    def load_maskrcnn(self, path):
        self.maskrcnn.load_state_dict(torch.load(open(path, 'rb')))

    def load_classification(self, path):
        feat, f1 = torch.load(open(path, 'rb'))
        self.features.load_state_dict(feat)
        self.f1.load_state_dict(f1)

    def load_transfer(self, path):
        feat, f1 = torch.load(open(path, 'rb'))
        self.maskrcnn.backbone.load_state_dict(feat)

    def load_all(self, path):
        self.load_state_dict(torch.load(open(path, 'rb')))

    def set_lr(self, lr):
        self.lr = lr
