"""
train_edge.py — 색칠판용 엣지 검출 모델(TEED) 학습
====================================================
데이터: BSDS500 (사람이 직접 그린 물체 경계, 이미지 500장)
  git clone --depth 1 https://github.com/BIDS/BSDS500.git ml/data/edge/bsds_repo
  - train 200장: 학습 / val 100장: 에폭 선택 / test 200장: 최종 평가

출력: ml/models/<name>.pt, ml/models/<name>.onnx → web/models/<name>.onnx (자동 복사, 기본 name=edge)

실행:
  pip install torch scipy opencv-python-headless onnx onnxruntime numpy
  python3 ml/train_edge.py              # 학습 + 평가 + 내보내기
  python3 ml/train_edge.py --eval-only  # 저장된 edge.pt로 평가 + 내보내기만
  python3 ml/train_edge.py --name edge_v2 --pos 0.5   # 다른 이름·라벨 기준으로 학습

정답 라벨: 여러 명의 주석을 평균 → POS_THRESH 이상 동의 = 경계(1), 일부만 동의 = 무시(2), 아무도 안 그음 = 0
  v1(edge): 30% 이상 + 정답 선 3x3 팽창  → 선이 굵고 질감선까지 학습
  v2(edge_v2): 50% 이상, 팽창 없음       → 사람 다수가 동의한 깔끔한 선만
"""

import os, sys, glob, time, shutil, argparse, random
import numpy as np
import cv2
import scipy.io
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_model import TED, EdgeNetForExport, RGB_MEAN, edge_prob

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BSDS_DIR  = 'ml/data/edge/bsds_repo/BSDS500/data'
MODEL_DIR = 'ml/models'
WEB_DIR   = 'web/models'
PT_PATH   = f'{MODEL_DIR}/edge.pt'      # --name으로 바뀜
ONNX_PATH = f'{MODEL_DIR}/edge.onnx'

CROP        = 320
BATCH       = 8
CROPS_PER_IMG = 4        # 한 에폭에 이미지당 랜덤 크롭 수
EPOCHS      = 40
LR          = 8e-4
WD          = 2e-4
POS_THRESH  = 0.5        # 주석자 동의 비율이 이 이상이면 경계 (--pos)
EVAL_GT     = (0.3, 0.5) # 평가 정답 기준 두 가지 (모델끼리 같은 기준으로 비교)
EVAL_TOL    = 2          # 평가 시 위치 허용 오차 (픽셀)

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


# ── 데이터 ────────────────────────────────────────────────
def load_split(split):
    """[(rgb uint8 HxWx3, agree float32 HxW)] — agree: 주석자 중 경계로 그은 비율"""
    out = []
    for img_path in sorted(glob.glob(f'{BSDS_DIR}/images/{split}/*.jpg')):
        name = os.path.splitext(os.path.basename(img_path))[0]
        rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        gts = scipy.io.loadmat(f'{BSDS_DIR}/groundTruth/{split}/{name}.mat')['groundTruth'][0]
        agree = np.mean([g['Boundaries'][0, 0].astype(np.float32) for g in gts], axis=0)
        out.append((rgb, agree))
    return out


def to_label(pos, anyline):
    """pos/anyline: 변형(리사이즈·회전) 후의 0~1 마스크. 선형 보간으로 흐려진 1px 선도 놓치지 않게 낮은 기준"""
    lab = np.zeros_like(pos, dtype=np.float32)
    lab[anyline > 0.05] = 2               # 일부만 그은 곳 → 손실에서 무시
    lab[pos > 0.3] = 1
    return lab


class EdgeTrainSet(Dataset):
    """웹캠 촬영 조건을 흉내 내는 증강: 크기·회전·뒤집기 + 밝기/대비/채도 + 블러 + 노이즈"""
    def __init__(self, items):
        # 경계 마스크(pos)와 누군가 그은 선(any)을 따로 변형 → 굵게 만들지 않고도 선이 안 끊김
        self.items = [(rgb, np.dstack([(agree >= POS_THRESH), (agree > 0)]).astype(np.float32))
                      for rgb, agree in items]

    def __len__(self):
        return len(self.items) * CROPS_PER_IMG

    def __getitem__(self, idx):
        rgb, agree = self.items[idx % len(self.items)]   # agree: HxWx2 (pos, any)
        h, w = agree.shape[:2]

        s = random.uniform(0.8, 1.5)
        ang = random.uniform(-12, 12)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
        nw, nh = int(w * s), int(h * s)
        M[0, 2] += (nw - w) / 2; M[1, 2] += (nh - h) / 2
        img = cv2.warpAffine(rgb, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ag  = cv2.warpAffine(agree, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        if nh < CROP or nw < CROP:   # 크롭보다 작으면 키움
            f = CROP / min(nh, nw) + 1e-3
            img = cv2.resize(img, None, fx=f, fy=f); ag = cv2.resize(ag, None, fx=f, fy=f)
            nh, nw = ag.shape[:2]
        y = random.randint(0, nh - CROP); x = random.randint(0, nw - CROP)
        img = img[y:y + CROP, x:x + CROP].astype(np.float32)
        ag  = ag[y:y + CROP, x:x + CROP]
        if random.random() < 0.5:
            img = img[:, ::-1]; ag = ag[:, ::-1]

        # 색·조명
        gray = img.mean(axis=2, keepdims=True)
        img = gray + (img - gray) * random.uniform(0.6, 1.4)                    # 채도
        img = (img - img.mean()) * random.uniform(0.7, 1.3) + img.mean()        # 대비
        img = img * random.uniform(0.7, 1.3) + random.uniform(-20, 20)          # 밝기
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 1.2))
        if random.random() < 0.5:
            img = img + np.random.normal(0, random.uniform(2, 8), img.shape)
        img = np.clip(img, 0, 255) - np.array(RGB_MEAN, np.float32)

        x_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
        y_t = torch.from_numpy(to_label(np.ascontiguousarray(ag[..., 0]), np.ascontiguousarray(ag[..., 1])))[None]
        return x_t, y_t


# ── 손실 (TEED 원본: 스케일 출력엔 가중 BCE, 융합 출력엔 CATS) ──────────
def weighted_bce(logits, label, l_w=1.0):
    pos = (label == 1).float(); neg = (label == 0).float()
    n_pos, n_neg = pos.sum(), neg.sum()
    w = pos * (n_neg / (n_pos + n_neg)) + neg * (1.1 * n_pos / (n_pos + n_neg))
    cost = F.binary_cross_entropy_with_logits(logits, pos, weight=w, reduction='none')
    return l_w * cost.mean((1, 2, 3)).sum()


def cats_loss(logits, label, tex_f=0.01, bdr_f=3.0, r=4):
    prob = torch.sigmoid(logits)
    edge = (label == 1).float()
    bce = weighted_bce(logits, label)

    k_r = torch.ones(1, 1, 2 * r + 1, 2 * r + 1, device=logits.device)
    k_3 = torch.ones(1, 1, 3, 3, device=logits.device)

    # 경계 추적: 정답 선 주변에서 예측이 선 위에 모이도록
    pred_bdr = edge * F.conv2d(prob * edge, k_r, padding=r)
    near = (F.conv2d(edge, k_r, padding=r) > 0).float() * (1 - edge)
    pred_tex = F.conv2d(prob * (1 - edge) * near, k_r, padding=r)
    sm = torch.clamp(pred_bdr / (pred_bdr + pred_tex + 1e-10), 1e-10, 1 - 1e-10)
    bdr = (-edge * torch.log(sm)).mean((1, 2, 3)).sum()

    # 질감 억제: 정답 선에서 먼 곳은 조용히
    far = 1 - (F.conv2d(edge, k_r, padding=r) > 0).float()
    tex = -torch.log(torch.clamp(1 - F.conv2d(prob, k_3, padding=1) / 9, 1e-10, 1 - 1e-10))
    tex = (tex * far).mean((1, 2, 3)).sum()

    return bce + bdr_f * bdr + tex_f * tex


SIDE_W = [1.1, 0.7, 1.1]


def total_loss(outs, label):
    return sum(weighted_bce(o, label, w) for o, w in zip(outs[:-1], SIDE_W)) + cats_loss(outs[-1], label)


# ── 추론 / 평가 ───────────────────────────────────────────
@torch.no_grad()
def predict(model, rgb):
    """rgb uint8 HxWx3 → 엣지 확률 HxW (8의 배수로 패딩 후 잘라냄)"""
    h, w = rgb.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    x = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT).astype(np.float32) - np.array(RGB_MEAN, np.float32)
    x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(DEVICE)
    return edge_prob(model(x)[-1])[0, 0, :h, :w].cpu().numpy()


def thin(prob):
    """간단한 비최대 억제: 가로 또는 세로 방향으로 극대인 픽셀만 남김"""
    p = np.pad(prob, 1, mode='edge')
    c = p[1:-1, 1:-1]
    hmax = (c >= p[1:-1, :-2]) & (c >= p[1:-1, 2:])
    vmax = (c >= p[:-2, 1:-1]) & (c >= p[2:, 1:-1])
    return prob * (hmax | vmax)


def f_score_curve(probs, agrees, thresholds, gt_thresh=None):
    """허용 오차 EVAL_TOL 안에서 맞으면 정답으로 치는 근사 경계 F-score (BSDS 공식 벤치마크의 간이판)"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EVAL_TOL + 1,) * 2)
    tp_p = np.zeros(len(thresholds)); n_p = np.zeros(len(thresholds))
    tp_r = np.zeros(len(thresholds)); n_g = 0
    for prob, agree in zip(probs, agrees):
        gt = (agree >= (gt_thresh or POS_THRESH)).astype(np.uint8)
        gt_d = cv2.dilate(gt, k).astype(bool)
        n_g += gt.sum()
        tp_ = thin(prob)
        for i, t in enumerate(thresholds):
            pred = (tp_ > t).astype(np.uint8)
            n_p[i] += pred.sum()
            tp_p[i] += (pred.astype(bool) & gt_d).sum()
            tp_r[i] += (gt.astype(bool) & cv2.dilate(pred, k).astype(bool)).sum()
    P = tp_p / np.maximum(n_p, 1); R = tp_r / max(n_g, 1)
    Fs = 2 * P * R / np.maximum(P + R, 1e-9)
    b = int(Fs.argmax())
    return Fs[b], P[b], R[b], thresholds[b]


def sobel_baseline(rgb):
    g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.5)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return mag / (mag.max() + 1e-9)


THRESHOLDS = np.linspace(0.05, 0.95, 19)


def evaluate(model, items, gt_thresh=None):
    model.eval()
    probs = [predict(model, rgb) for rgb, _ in items]
    return f_score_curve(probs, [a for _, a in items], THRESHOLDS, gt_thresh)


# ── 내보내기 ──────────────────────────────────────────────
def export_onnx(model):
    import onnxruntime as ort
    net = EdgeNetForExport(model.cpu().eval())
    dummy = torch.rand(1, 3, 360, 480) * 255
    torch.onnx.export(net, dummy, ONNX_PATH, input_names=['rgb'], output_names=['edge'],
                      dynamic_axes={'rgb': {2: 'h', 3: 'w'}, 'edge': {2: 'h', 3: 'w'}},
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(ONNX_PATH)
    got = sess.run(None, {'rgb': dummy.numpy()})[0]
    with torch.no_grad():
        ref = net(dummy).numpy()
    print(f'  ONNX 검증: 최대 오차 {np.abs(got - ref).max():.2e}, 크기 {os.path.getsize(ONNX_PATH) / 1024:.0f}KB')
    os.makedirs(WEB_DIR, exist_ok=True)
    dst = f'{WEB_DIR}/{os.path.basename(ONNX_PATH)}'
    shutil.copy(ONNX_PATH, dst)
    print(f'  → {dst} 복사 완료')


# ── 메인 ──────────────────────────────────────────────────
def main():
    global PT_PATH, ONNX_PATH, POS_THRESH
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--name', default='edge', help='저장 파일 이름 (ml/models/<name>.pt/.onnx)')
    ap.add_argument('--pos', type=float, default=POS_THRESH, help='정답 경계 기준 (주석자 동의 비율)')
    args = ap.parse_args()
    PT_PATH, ONNX_PATH = f'{MODEL_DIR}/{args.name}.pt', f'{MODEL_DIR}/{args.name}.onnx'
    POS_THRESH = args.pos
    print(f'모델 이름: {args.name}, 정답 기준: {POS_THRESH}')

    if not os.path.isdir(BSDS_DIR):
        sys.exit(f'BSDS500이 없습니다: {BSDS_DIR}\n  git clone --depth 1 https://github.com/BIDS/BSDS500.git ml/data/edge/bsds_repo')

    print(f'장치: {DEVICE}')
    t0 = time.time()
    train, val, test = load_split('train'), load_split('val'), load_split('test')
    print(f'데이터: train {len(train)} / val {len(val)} / test {len(test)}  ({time.time() - t0:.1f}s)')

    model = TED().to(DEVICE)
    print(f'파라미터: {sum(p.numel() for p in model.parameters()):,}')

    if not args.eval_only:
        loader = DataLoader(EdgeTrainSet(train), batch_size=BATCH, shuffle=True, num_workers=4,
                            drop_last=True, persistent_workers=True)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=args.epochs * len(loader),
                                                    pct_start=0.1)
        best = -1
        for ep in range(args.epochs):
            model.train(); t = time.time(); losses = []
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = total_loss(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                losses.append(loss.item())
            Fv, Pv, Rv, tv = evaluate(model, val)
            mark = ''
            if Fv > best:
                best = Fv; torch.save(model.state_dict(), PT_PATH); mark = '  ★ 저장'
            print(f'[{ep + 1:2d}/{args.epochs}] loss {np.mean(losses):.4f}  val F {Fv:.3f} '
                  f'(P {Pv:.3f} R {Rv:.3f} @ {tv:.2f})  {time.time() - t:.0f}s{mark}', flush=True)

    model.load_state_dict(torch.load(PT_PATH, map_location=DEVICE))
    print('\n── 테스트셋 (200장, 허용오차 2px 근사 F-score) ──')
    sobel = [sobel_baseline(r) for r, _ in test]
    for g in EVAL_GT:
        Ft, Pt, Rt, tt = evaluate(model, test, g)
        Fs, Ps, Rs, _ = f_score_curve(sobel, [a for _, a in test], THRESHOLDS, g)
        print(f'  [정답 = 주석자 {int(g * 100)}% 이상 동의]')
        print(f'    {args.name:14s}: F {Ft:.3f}  (P {Pt:.3f} R {Rt:.3f}, 임계값 {tt:.2f})')
        print(f'    Sobel 기준선  : F {Fs:.3f}  (P {Ps:.3f} R {Rs:.3f})')

    export_onnx(model)


if __name__ == '__main__':
    main()
