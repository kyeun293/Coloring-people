"""
edge_datasets.py: 통일 데이터셋을 바로 학습과 평가에 쓰는 PyTorch Dataset과 라벨 (담당: Taein)

먼저 데이터 만들기: python edge/data/unified_data.py

A (edge detection 학습):
    import sys; sys.path.insert(0, 'edge/data')
    from edge_datasets import EdgeDataset, rcf_loss
    from torch.utils.data import DataLoader

    train = EdgeDataset('train', datasets=('bsds', 'pascal'), augment=True, crop=320, norm='bgr_mean')
    loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=2)
    for b in loader:
        logits = model(b['image'])                      # (B,3,320,320)
        loss = rcf_loss(logits, b['edge'], b['valid'])  # 표준 RCF 손실 (원하면 직접 만든 손실 사용)

    val = EdgeDataset('val', datasets=('bsds',), norm='bgr_mean')   # augment 없음, 원본 크기 → batch_size=1
    for b in DataLoader(val, batch_size=1):
        prob = torch.sigmoid(model(b['image']))         # b['bnd']: 주석자별 경계 GT (평가용)

B (segmentation 튜닝/평가, 학습 없음):
    from edge_datasets import iter_eval
    for s in iter_eval('val', 'bsds'):
        s['image'], s['seg'], s['bnd'], s['id']          # seg: (주석자,H,W) 구역 GT / bnd: 경계 GT

반환 dict (EdgeDataset):
    image  (3,H,W) float32   norm 옵션대로 정규화
    edge   (1,H,W) float32   학습 라벨: 그 픽셀에 경계를 그은 주석자 비율 (0~1). PASCAL은 0 또는 1
    valid  (1,H,W) float32   1 = 실제 이미지, 0 = crop 때 채운 빈 칸 (손실에서 제외)
    bnd    (K,H,W) bool      augment=False일 때만: 주석자별 경계 GT (평가용)
    seg    (K,H,W) int32     augment=False일 때만: 주석자별 구역 GT
    id, dataset              이미지 id, 'bsds' | 'pascal'
"""
import os, random, sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_data import UNI, list_ids, load_sample, seg2bdry

# 모델별 입력 정규화 (MODELS.md 참고)
NORMS = {
    'bgr_mean': dict(rgb=False, mean=[103.939, 116.779, 123.68], std=[1, 1, 1], scale=1.0),         # HED, TEED, DexiNed
    'imagenet_rgb': dict(rgb=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=1 / 255),  # PiDiNet
    None: dict(rgb=False, mean=[0, 0, 0], std=[1, 1, 1], scale=1 / 255),                           # BGR 0~1 그대로
}


# ── 라벨 ────────────────────────────────────────────────
def edge_label(bnd):
    """주석자별 경계 (K,H,W) bool → 학습 라벨 (H,W) float32: 경계를 그은 주석자 비율 (HED/RCF 관례)"""
    return np.asarray(bnd, np.float32).mean(0)


def rcf_target(edge, valid, eta=0.5):
    """RCF(Liu et al., CVPR 2017) 라벨: 주석자 비율 ≥ eta → 경계(1), 0 → 배경(0), 그 사이(애매함)와 빈 칸 → 무시.
    반환: (target, mask) 모두 edge와 같은 형태. mask=0인 픽셀은 손실에서 제외"""
    target = (edge >= eta).float()
    mask = valid * ((edge == 0) | (edge >= eta)).float()
    return target, mask


def rcf_loss(logits, edge, valid, eta=0.5, lam=1.1):
    """RCF의 클래스 균형 binary cross-entropy. logits가 여러 개(side outputs)면 모두 더함.
    경계 픽셀이 아주 적어서(약 2%) 경계 쪽 가중치를 크게 줌: w+ = |Y-|/|Y|, w- = lam·|Y+|/|Y| (이미지마다)"""
    outs = logits if isinstance(logits, (list, tuple)) else [logits]
    target, mask = rcf_target(edge, valid, eta)
    pos = (target * mask).sum((1, 2, 3), keepdim=True)
    neg = ((1 - target) * mask).sum((1, 2, 3), keepdim=True)
    tot = (pos + neg).clamp(min=1)
    weight = mask * torch.where(target > 0, neg / tot, lam * pos / tot)
    loss = 0
    for o in outs:
        if o.shape[-2:] != edge.shape[-2:]:
            o = torch.nn.functional.interpolate(o, edge.shape[-2:], mode='bilinear', align_corners=False)
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(o, target, weight, reduction='sum')
    return loss / edge.shape[0]


# ── augmentation (train만) ──────────────────────────────
def _resize_bnd(seg, size):
    """크기를 바꾼 경계 GT: 구역 지도를 최근접으로 바꾼 뒤 seg2bdry로 경계를 다시 그림
    (경계를 직접 보간하면 1픽셀 선이 두꺼워지거나 끊겨서). GT를 만든 규칙과 같은 방식"""
    return np.stack([seg2bdry(cv2.resize(s, size, interpolation=cv2.INTER_NEAREST)) for s in seg])


def augment_sample(img, seg, bnd, scale_range=(0.5, 1.5), rng=random):
    """HED/RCF 관례를 따른 augmentation: 크기 변환 + 뒤집기 + 회전.
    회전은 90° 단위만 써요 (임의 각도는 보간 때문에 1픽셀 경계가 흐려지고 모서리가 비어서).
    크기를 바꾸지 않으면 원본 경계 GT를 그대로 써요."""
    s = rng.uniform(*scale_range)
    if abs(s - 1) > 1e-3:
        size = (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s)))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        bnd = _resize_bnd(seg, size)
    if rng.random() < 0.5:
        img, bnd = img[:, ::-1], bnd[:, :, ::-1]
    k = rng.randrange(4)
    if k:
        img, bnd = np.rot90(img, k, (0, 1)), np.rot90(bnd, k, (1, 2))
    return np.ascontiguousarray(img), np.ascontiguousarray(bnd)


def random_crop(img, edge, size, rng=random):
    """size×size로 자름. 이미지가 더 작으면 빈 칸을 채우고 valid=0으로 표시"""
    h, w = edge.shape
    H, W = max(h, size), max(w, size)
    img_p = np.zeros((H, W, 3), img.dtype); img_p[:h, :w] = img
    edge_p = np.zeros((H, W), np.float32); edge_p[:h, :w] = edge
    valid = np.zeros((H, W), np.float32); valid[:h, :w] = 1
    y, x = rng.randrange(H - size + 1), rng.randrange(W - size + 1)
    sl = (slice(y, y + size), slice(x, x + size))
    return img_p[sl], edge_p[sl], valid[sl]


def normalize(img, norm):
    p = NORMS[norm]
    x = img[..., ::-1] if p['rgb'] else img
    x = (x.astype(np.float32) * p['scale'] - p['mean']) / p['std']
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32))


# ── Dataset ─────────────────────────────────────────────
class EdgeDataset(Dataset):
    """
    split:     'train' | 'val' | 'test'
    datasets:  ('bsds',), ('pascal',), ('bsds', 'pascal')  여러 개면 이어 붙임
    augment:   True면 크기 변환, 뒤집기, 90° 회전 + crop (train용). False면 원본 크기 그대로 + 평가용 GT 포함
    crop:      augment=True일 때 자를 크기 (배치로 묶으려면 필요). None이면 자르지 않음
    norm:      'bgr_mean' (HED, TEED, DexiNed) | 'imagenet_rgb' (PiDiNet) | None (BGR 0~1)
    repeat:    {'bsds': 10} 처럼 데이터셋별 반복 횟수 (BSDS 200장 vs PASCAL 2,499장 균형 맞추기)
    """

    def __init__(self, split, datasets=('bsds', 'pascal'), augment=None, crop=320, norm='bgr_mean',
                 repeat=None, scale_range=(0.5, 1.5), root=UNI):
        self.split, self.root, self.norm = split, root, norm
        self.augment = (split == 'train') if augment is None else augment
        self.crop = crop if self.augment else None
        self.scale_range = scale_range
        repeat = repeat or {}
        self.items = [(d, i) for d in datasets for _ in range(repeat.get(d, 1)) for i in list_ids(d, split, root)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        d, img_id = self.items[idx]
        img, seg, bnd = load_sample(d, self.split, img_id, self.root)
        out = dict(id=img_id, dataset=d)
        if self.augment:
            img, bnd_a = augment_sample(img, seg, bnd, self.scale_range)
            edge = edge_label(bnd_a)
            if self.crop:
                img, edge, valid = random_crop(img, edge, self.crop)
            else:
                valid = np.ones_like(edge)
        else:
            edge, valid = edge_label(bnd), np.ones(bnd.shape[1:], np.float32)
            out['bnd'] = torch.from_numpy(bnd)
            out['seg'] = torch.from_numpy(seg)
        out['image'] = normalize(img, self.norm)
        out['edge'] = torch.from_numpy(np.ascontiguousarray(edge))[None]
        out['valid'] = torch.from_numpy(np.ascontiguousarray(valid))[None]
        return out


def iter_eval(split, dataset, root=UNI):
    """B용: 평가할 이미지를 차례로 줌. image는 BGR uint8 원본"""
    for img_id in list_ids(dataset, split, root):
        img, seg, bnd = load_sample(dataset, split, img_id, root)
        yield dict(id=img_id, dataset=dataset, image=img, seg=seg, bnd=bnd)
