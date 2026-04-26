"""
55230316 基线模型 - 模型定义 + 训练脚本
用法：
    python 55230316_baseline_train.py
    python 55230316_baseline_train.py --epochs 5 --batch 16 --lr 1e-4
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from PIL import Image

# ── 常量 ──────────────────────────────────────────────────────────────────────
CLASS_NORMALIZE = {
    'car': 'car', 'truck': 'truck', 'bus': 'bus', 'van': 'van',
    'feright_car': 'truck', 'feright car': 'truck', 'feright': 'truck',
    'truvk': 'truck',
}
STANDARD_CLASSES = {'bus': 0, 'car': 1, 'truck': 2, 'van': 3}
IMG_SIZE = 640


# ── 数据集 ────────────────────────────────────────────────────────────────────
def parse_xml_annotation(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
        size = root.find('size')
        W = int(size.find('width').text)
        H = int(size.find('height').text)
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            poly = obj.find('polygon')
            if poly is None:
                continue
            xs = [int(poly.find(f'x{i}').text) for i in range(1, 5)]
            ys = [int(poly.find(f'y{i}').text) for i in range(1, 5)]
            objects.append({
                'class': name,
                'bbox': [min(xs), min(ys), max(xs), max(ys)],
            })
        return {'image_width': W, 'image_height': H, 'objects': objects}
    except Exception:
        return None


class DroneVehicleDataset(Dataset):
    """单模态（RGB only）DroneVehicle 数据集"""

    def __init__(self, dataset_root, split='train', img_size=IMG_SIZE):
        self.dataset_root = Path(dataset_root)
        self.img_size = img_size

        if split == 'train':
            self.img_dir   = self.dataset_root / 'train' / 'trainimg'
            self.label_dir = self.dataset_root / 'train' / 'trainlabel'
        else:
            self.img_dir   = self.dataset_root / split / f'{split}img'
            self.label_dir = self.dataset_root / split / f'{split}label'

        self.img_files    = sorted(self.img_dir.glob('*.jpg'))
        self.class_to_idx = STANDARD_CLASSES
        self.idx_to_class = {v: k for k, v in STANDARD_CLASSES.items()}

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path   = self.img_files[idx]
        label_path = self.label_dir / (img_path.stem + '.xml')

        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((self.img_size, self.img_size))
        img_tensor = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)  # (3, H, W)

        anno = parse_xml_annotation(label_path)
        targets = []
        if anno:
            for obj in anno['objects']:
                cls_norm = CLASS_NORMALIZE.get(obj['class'])
                if cls_norm is None:
                    continue
                xmin, ymin, xmax, ymax = obj['bbox']
                cx = (xmin + xmax) / 2.0 / orig_w
                cy = (ymin + ymax) / 2.0 / orig_h
                w  = (xmax - xmin) / orig_w
                h  = (ymax - ymin) / orig_h
                targets.append([self.class_to_idx[cls_norm], cx, cy, w, h])

        return {'image': img_tensor, 'targets': targets, 'image_path': str(img_path)}


def collate_fn(batch):
    images      = torch.stack([b['image'] for b in batch])
    image_paths = [b['image_path'] for b in batch]
    max_t = max(len(b['targets']) for b in batch)
    max_t = max(max_t, 1)
    padded = []
    for b in batch:
        t = b['targets']
        t = t + [[-1, -1, -1, -1, -1]] * (max_t - len(t))
        padded.append(torch.tensor(t, dtype=torch.float32))
    return {
        'image':      images,
        'targets':    torch.stack(padded),
        'image_path': image_paths,
    }


# ── 模型 ──────────────────────────────────────────────────────────────────────
class BaselineDetector(nn.Module):
    """ResNet18 backbone + 单尺度检测头（3 anchors，RGB only）"""

    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        backbone = resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # (B,512,H,W)
        self.det_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, (num_classes + 5) * 3, 1),
        )

    def forward(self, x):
        feat = self.backbone(x)
        raw  = self.det_head(feat)
        B    = x.size(0)
        h_f, w_f = feat.size(2), feat.size(3)
        raw = raw.view(B, 3, self.num_classes + 5, h_f * w_f)

        # grid 中心点
        cx_r = (torch.arange(w_f, device=x.device).float() + 0.5) / w_f
        cy_r = (torch.arange(h_f, device=x.device).float() + 0.5) / h_f
        gcx, gcy = torch.meshgrid(cx_r, cy_r, indexing='ij')
        gcx = gcx.flatten().view(1, 1, 1, -1)
        gcy = gcy.flatten().view(1, 1, 1, -1)

        pred_cx   = (torch.sigmoid(raw[:, :, 0:1, :]) * 2 - 0.5 + gcx).clamp(0, 1)
        pred_cy   = (torch.sigmoid(raw[:, :, 1:2, :]) * 2 - 0.5 + gcy).clamp(0, 1)
        pred_w    = torch.sigmoid(raw[:, :, 2:3, :])
        pred_h    = torch.sigmoid(raw[:, :, 3:4, :])
        pred_conf = raw[:, :, 4:5, :]   # raw logit
        pred_cls  = raw[:, :, 5:,  :]   # raw logit

        return torch.cat([pred_cx, pred_cy, pred_w, pred_h, pred_conf, pred_cls], dim=2)


# ── 损失函数 ──────────────────────────────────────────────────────────────────
class DetectionLoss(nn.Module):
    """Grid Matching Loss（正负样本分离）"""

    def __init__(self, num_classes=4, lam_conf=1.0, lam_loc=5.0, lam_cls=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.lam_conf = lam_conf
        self.lam_loc  = lam_loc
        self.lam_cls  = lam_cls
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.sl1 = nn.SmoothL1Loss(reduction='sum')
        self.ce  = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, preds, targets):
        """
        preds  : (B, 3, 5+C, hw)  — cx/cy/w/h decoded, conf/cls raw logit
        targets: (B, max_T, 5)    — [cls, cx, cy, w, h] 归一化，-1 为 padding
        """
        B, num_a, _, hw = preds.shape
        h_f = int(hw ** 0.5)
        w_f = hw // h_f

        pred_bbox = preds[:, :, :4, :].permute(0, 1, 3, 2).reshape(B, -1, 4)
        pred_conf = preds[:, :, 4,  :].reshape(B, -1)
        pred_cls  = preds[:, :, 5:, :].permute(0, 1, 3, 2).reshape(B, -1, self.num_classes)

        gcx = (torch.arange(w_f, device=preds.device).float() + 0.5) / w_f
        gcy = (torch.arange(h_f, device=preds.device).float() + 0.5) / h_f
        gcx, gcy = torch.meshgrid(gcx, gcy, indexing='ij')
        gcx = gcx.flatten().unsqueeze(0).expand(num_a, -1).flatten()
        gcy = gcy.flatten().unsqueeze(0).expand(num_a, -1).flatten()

        tp_conf = tn_conf = t_loc = t_cls = 0.0
        n_pos = n_neg = 0

        for b in range(B):
            valid = targets[b, :, 0] >= 0
            gt = targets[b, valid]

            if len(gt) == 0:
                tn_conf += self.bce(pred_conf[b], torch.zeros_like(pred_conf[b])).sum()
                n_neg += pred_conf.size(1)
                continue

            gt_cx, gt_cy = gt[:, 1], gt[:, 2]
            gt_w,  gt_h  = gt[:, 3], gt[:, 4]
            gt_cls = gt[:, 0].long()
            gt_box = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=1)

            dist = (gcx.unsqueeze(0) - gt_cx.unsqueeze(1)).abs() + \
                   (gcy.unsqueeze(0) - gt_cy.unsqueeze(1)).abs()
            best = dist.argmin(dim=1)

            n_pos += len(gt)
            tp_conf += self.bce(pred_conf[b, best], torch.ones(len(gt), device=preds.device)).sum()

            neg_mask = torch.ones(pred_conf.size(1), dtype=torch.bool, device=preds.device)
            neg_mask[best] = False
            tn_conf += self.bce(pred_conf[b, neg_mask], torch.zeros(neg_mask.sum(), device=preds.device)).sum()
            n_neg += neg_mask.sum().item()

            t_loc += self.sl1(pred_bbox[b, best], gt_box)
            t_cls += self.ce(pred_cls[b, best], gt_cls)

        l_conf = tp_conf / max(n_pos, 1) + 0.5 * tn_conf / max(n_neg, 1)
        l_loc  = t_loc / max(n_pos, 1)
        l_cls  = t_cls / max(n_pos, 1)
        return self.lam_conf * l_conf + self.lam_loc * l_loc + self.lam_cls * l_cls


# ── mAP 评估 ──────────────────────────────────────────────────────────────────
def _iou(b1, b2):
    def xyxy(b): return b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    x1,y1,x2,y2 = xyxy(b1); x3,y3,x4,y4 = xyxy(b2)
    ix = max(0, min(x2,x4)-max(x1,x3)); iy = max(0, min(y2,y4)-max(y1,y3))
    inter = ix * iy
    union = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter
    return inter / (union + 1e-6)


def _ap(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    if not pred_boxes or not gt_boxes:
        return 0.0
    order   = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
    tp = np.zeros(len(order)); fp = np.zeros(len(order))
    matched = [False] * len(gt_boxes)
    for rank, i in enumerate(order):
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if not matched[j]:
                iou = _iou(pred_boxes[i], gb)
                if iou > best_iou: best_iou, best_j = iou, j
        if best_iou >= iou_thr: tp[rank] = 1; matched[best_j] = True
        else: fp[rank] = 1
    cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
    prec = cum_tp / (cum_tp + cum_fp + 1e-6)
    rec  = cum_tp / (len(gt_boxes) + 1e-6)
    return sum(np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0
               for t in np.linspace(0, 1, 11)) / 11


def evaluate_map(model, loader, num_classes, device, iou_thr=0.5, conf_thr=0.1):
    all_pred = {c: {'boxes': [], 'scores': []} for c in range(num_classes)}
    all_gt   = {c: [] for c in range(num_classes)}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs    = batch['image'].to(device)
            targets = batch['targets'].to(device)
            preds   = model(imgs)
            for b in range(imgs.size(0)):
                for gt in targets[b, targets[b, :, 0] >= 0]:
                    all_gt[int(gt[0].item())].append(gt[1:5].cpu().tolist())
                for a in range(preds.size(1)):
                    pa    = preds[b, a]
                    conf  = torch.sigmoid(pa[4])
                    cls_s = torch.softmax(pa[5:], dim=0)
                    score = conf * cls_s.max(0).values
                    cls_i = cls_s.argmax(0)
                    for i in (score > conf_thr).nonzero(as_tuple=True)[0]:
                        c = int(cls_i[i].item())
                        cx, cy, w, h = pa[0,i].item(), pa[1,i].item(), pa[2,i].item(), pa[3,i].item()
                        if w < 0.005 or h < 0.005: continue
                        all_pred[c]['boxes'].append([cx, cy, w, h])
                        all_pred[c]['scores'].append(score[i].item())
    ap_list = [_ap(all_pred[c]['boxes'], all_pred[c]['scores'], all_gt[c], iou_thr)
               for c in range(num_classes)]
    return float(np.mean(ap_list)), ap_list


# ── 训练主函数 ────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    dataset_root = Path(args.data)
    train_ds = DroneVehicleDataset(dataset_root, split='train')
    val_ds   = DroneVehicleDataset(dataset_root, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, drop_last=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=4, collate_fn=collate_fn)
    print(f'Train: {len(train_ds)} images, Val: {len(val_ds)} images')

    model     = BaselineDetector(num_classes=4).to(device)
    loss_fn   = DetectionLoss(num_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = defaultdict(list)
    best_map = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        total_loss, n_batch = 0.0, 0
        for i, batch in enumerate(train_loader):
            imgs = batch['image'].to(device)
            tgts = batch['targets'].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), tgts)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item(); n_batch += 1
            if (i + 1) % 50 == 0:
                print(f'Epoch {epoch}/{args.epochs}  Batch {i+1}/{len(train_loader)}  Loss: {loss.item():.4f}')
        scheduler.step()

        # ── val loss ──
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += loss_fn(model(batch['image'].to(device)),
                                    batch['targets'].to(device)).item()
                n_val += 1

        avg_train = total_loss / n_batch
        avg_val   = val_loss / max(n_val, 1)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        # ── mAP ──
        mAP, ap_list = evaluate_map(model, val_loader, 4, device)
        history['mAP'].append(mAP)
        idx_to_class = {v: k for k, v in STANDARD_CLASSES.items()}
        print(f'\nEpoch {epoch} | train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  mAP@0.5={mAP:.4f}')
        for c, ap in enumerate(ap_list):
            print(f'  {idx_to_class[c]:8s}: {ap:.4f}')

        if mAP > best_map:
            best_map = mAP
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'mAP': mAP, 'history': dict(history)},
                       save_dir / 'best.pth')
            print(f'  → saved best.pth (mAP={mAP:.4f})')

    torch.save({'epoch': args.epochs, 'model': model.state_dict(),
                'history': dict(history)}, save_dir / 'last.pth')
    print(f'\nDone. Best mAP@0.5 = {best_map:.4f}')
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     default='../datasets/DroneVehicle')
    parser.add_argument('--epochs',   type=int,   default=3)
    parser.add_argument('--batch',    type=int,   default=16)
    parser.add_argument('--lr',       type=float, default=1e-4)
    parser.add_argument('--save_dir', default='checkpoints/baseline')
    args = parser.parse_args()
    train(args)
