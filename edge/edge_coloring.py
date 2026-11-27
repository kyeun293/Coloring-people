# %% [markdown]
# # Edge Detection 기반 색칠공부판 생성
#
# **담당: Taein** · Computer Vision 팀 프로젝트 (coloring-people)
#
# 사진 → **edge 검출** → **후처리(닫힌 구역 만들기)** → 색칠공부판 + 팔레트
#
# | 단계 | 내용 |
# |---|---|
# | 0 | 설정 / 데이터(BSDS500) / HED 모델 다운로드 |
# | 1 | Edge 검출기: Canny, Sobel(강의), HED(딥러닝), k-means(팀 원본 방식, 비교 기준) |
# | 2 | 후처리: 이진화 → 얇게 → 끊긴 선 잇기 → 구역 분할 → 작은 구역 병합 → 경계선 |
# | 3 | 색칠공부판 + 팔레트 생성 |
# | 4 | 평가: Boundary F, **Leakage(색 새는 정도)**, Fragmentation, 구역 수, 시간 |
# | 5 | 후처리 파라미터 실험 (ablation) |
# | 6 | 내 사진 / 웹캠으로 해보기 |
#
# 무료 Colab CPU만으로 돌아가요. (`런타임 → 모두 실행`)

# %%
# ── 0-1. 라이브러리 & 경로 설정 ─────────────────────────────
import os, sys, time, subprocess, urllib.request
from functools import partial
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import io as sio
from skimage.morphology import skeletonize
from skimage.measure import label as sk_label

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    EDGE_DIR = '/content'
elif '__file__' in globals():
    EDGE_DIR = os.path.dirname(os.path.abspath(__file__))     # python edge/edge_coloring.py
else:
    EDGE_DIR = os.path.abspath('.')                            # 로컬 Jupyter: edge/ 에서 실행했다고 가정
WORK   = os.environ.get('EDGE_WORK', os.path.join(EDGE_DIR, 'edge_work'))   # 데이터·모델 (용량 큼)
DATA   = os.path.join(WORK, 'data')
MODELS = os.path.join(WORK, 'models')
OUT    = os.environ.get('EDGE_OUT', os.path.join(EDGE_DIR, 'results'))      # 결과 그림·CSV는 항상 여기
for d in (DATA, MODELS, OUT):
    os.makedirs(d, exist_ok=True)

N_EVAL = int(os.environ.get('N_EVAL', 30))   # 평가에 쓸 테스트 이미지 수 (최대 200)
RUN_DEMO = globals().get('RUN_DEMO', True)   # 예시 그림 셀 실행 여부 (스크립트에서 불러올 때는 False)
SEED = 0
np.random.seed(SEED)
print('Colab' if IN_COLAB else 'Local', '| 데이터:', WORK, '| 결과:', OUT, '| OpenCV', cv2.__version__)


def show(images, titles=None, cols=None, size=4, save=None):
    """여러 이미지를 격자로 표시 (컬러는 RGB로 넘길 것). 그래프 글자는 Colab 폰트 문제로 영어 사용"""
    n = len(images)
    cols = cols or n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows * 0.75))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, range(len(axes))):
        ax.axis('off')
        if i >= n:
            continue
        im = images[i]
        ax.imshow(im, cmap=None if im.ndim == 3 else 'gray')
        if titles:
            ax.set_title(titles[i], fontsize=11)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUT, save), dpi=90)
    plt.show()
    plt.close(fig)


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# %%
# ── 0-2. 데이터: BSDS500 (경계 GT + 구역 분할 GT, 테스트 200장) ─────
BSDS_DIR = os.path.join(DATA, 'BSDS500')
if not os.path.isdir(BSDS_DIR):
    print('BSDS500 다운로드 중... (약 1분)')
    subprocess.run(['git', 'clone', '-q', '--depth', '1',
                    'https://github.com/BIDS/BSDS500.git', BSDS_DIR], check=True)

BSDS_DATA = os.path.join(BSDS_DIR, 'BSDS500', 'data')
IMG_DIR = os.path.join(BSDS_DATA, 'images', 'test')
GT_DIR  = os.path.join(BSDS_DATA, 'groundTruth', 'test')


def list_ids(split):
    d = os.path.join(BSDS_DATA, 'images', split)
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith('.jpg'))


test_ids = list_ids('test')   # 최종 평가용 (마지막에 한 번만)
val_ids  = list_ids('val')    # 개발용 소규모 셋은 여기서 고름 (edge/dev_set.json)
print('test', len(test_ids), '장 / val', len(val_ids), '장')


def load_image(img_id, split='test'):
    return cv2.imread(os.path.join(BSDS_DATA, 'images', split, img_id + '.jpg'))   # BGR


def load_gt(img_id, split='test'):
    """사람 주석자(보통 5명)마다 구역 분할(Segmentation)과 경계(Boundaries) 하나씩"""
    gts = sio.loadmat(os.path.join(BSDS_DATA, 'groundTruth', split, img_id + '.mat'))['groundTruth'][0]
    segs = [g['Segmentation'][0, 0].astype(np.int32) for g in gts]
    bnds = [g['Boundaries'][0, 0].astype(bool) for g in gts]
    return segs, bnds


sample_id = test_ids[0]
if RUN_DEMO:
    _segs, _bnds = load_gt(sample_id)
    show([bgr2rgb(load_image(sample_id)), _bnds[0], np.any(_bnds, 0)],
         ['image', 'GT boundary (1 annotator)', f'GT boundary ({len(_bnds)} annotators)'])

# %%
# ── 0-3. HED 사전학습 모델 (Caffe, OpenCV dnn으로 CPU 추론) ─────
HED_PROTO = os.path.join(MODELS, 'hed_deploy.prototxt')
HED_MODEL = os.path.join(MODELS, 'hed_pretrained_bsds.caffemodel')
HED_URLS = {
    HED_PROTO: 'https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt',
    HED_MODEL: 'https://raw.githubusercontent.com/ashukid/hed-edge-detector/master/hed_pretrained_bsds.caffemodel',
}
for path, url in HED_URLS.items():
    if not os.path.exists(path):
        print('다운로드:', os.path.basename(path))
        urllib.request.urlretrieve(url, path)


class CropLayer:
    """HED의 Crop 레이어 (OpenCV dnn에 직접 등록해야 함): 업샘플된 side output을 입력 크기로 자름"""
    def __init__(self, params, blobs):
        self.ys = self.ye = self.xs = self.xe = 0

    def getMemoryShapes(self, inputs):
        in_shape, target = inputs[0], inputs[1]
        b, c = in_shape[0], in_shape[1]
        h, w = target[2], target[3]
        self.ys = (in_shape[2] - h) // 2
        self.xs = (in_shape[3] - w) // 2
        self.ye, self.xe = self.ys + h, self.xs + w
        return [[b, c, h, w]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ys:self.ye, self.xs:self.xe]]


try:
    cv2.dnn_registerLayer('Crop', CropLayer)
except cv2.error:
    pass   # 셀을 다시 실행해서 이미 등록된 경우
hed_net = cv2.dnn.readNetFromCaffe(HED_PROTO, HED_MODEL)
print('HED 로드 완료')

# %% [markdown]
# ## 1. Edge 검출기
#
# 모든 검출기는 **edge 확률 맵**(0~1, 이미지와 같은 크기)을 반환하고, 그 뒤의 후처리는 공통으로 적용해요.
#
# - **Canny**: 블러 → 기울기 → NMS → 이중 임계값. 얇고 정확하지만 질감(잔디, 무늬)까지 잡아요.
# - **Sobel**: 기울기 크기만 사용. 강의 내용으로 직접 바꿔 볼 수 있도록 커널 컨볼루션을 그대로 적었어요.
# - **HED**: 사람이 그린 경계를 학습한 딥러닝 모델. 물체 윤곽 위주로 잡지만 선이 두꺼워요.
# - **k-means (팀 원본 방식)**: edge 없이 색을 K개로 나누어 구역을 만드는 방식. 비교 기준(baseline)이에요.

# %%
def detect_canny(img, sigma=0.33, blur=5):
    """중앙값 기반 자동 임계값 Canny"""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (blur, blur), 0)
    v = float(np.median(gray))
    lo, hi = int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v))
    return cv2.Canny(gray, lo, hi).astype(np.float32) / 255.0


def detect_sobel(img, blur=5):
    """Sobel 기울기 크기 (강의 내용대로 직접 구현해서 바꿔 보세요: NMS, 다른 커널 등)"""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (blur, blur), 0).astype(np.float32)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gx = cv2.filter2D(gray, -1, kx)
    gy = cv2.filter2D(gray, -1, kx.T)
    mag = np.hypot(gx, gy)
    return np.clip(mag / (np.percentile(mag, 99) + 1e-6), 0, 1)


HED_MEAN = (104.00698793, 116.66876762, 122.67891434)

def detect_hed(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (w, h), HED_MEAN, swapRB=False, crop=False)
    hed_net.setInput(blob)
    return hed_net.forward()[0, 0].astype(np.float32)


DETECTORS = {'canny': detect_canny, 'sobel': detect_sobel, 'hed': detect_hed}

# %% [markdown]
# ### 1-2. 추가 사전학습 모델 (학습 없이 공개 가중치로 추론만)
#
# | 모델 | 학습 데이터 | 크기 | 출처 |
# |---|---|---|---|
# | **PiDiNet** (ICCV 2021) | BSDS500 (`table5_pidinet.pth`) | 약 2.8MB | github.com/hellozhuo/pidinet |
# | **TEED** (ICCV-W 2023) | **BIPED** (BSDS 아님) | 약 0.25MB | github.com/xavysp/TEED |
# | **DexiNed** (WACV 2020) | **BIPED** (BSDS 아님) | 약 135MB | `kornia` 내장 |
#
# TEED·DexiNed는 원 저자 코드처럼 sigmoid 뒤 이미지별 최소~최대 정규화를 해요 (출력값이 보정돼 있지 않음).
# 후처리 파라미터는 튜닝 없이 모두 HED 기본값(thresh=0.3 등)을 그대로 써요.
#
# ⚠️ HED와 PiDiNet은 BSDS500(train/val)로 학습된 모델이라 val(dev set) 점수가 실제보다 좋게 나올 수 있어요.
# TEED와 DexiNed는 BSDS를 본 적이 없어서 어느 split에서든 공정해요. test는 모든 모델에 공정해요.

# %%
import importlib
import torch

TORCH_DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def clone_repo(url, name):
    path = os.path.join(MODELS, name)
    if not os.path.isdir(path):
        print('다운로드:', name)
        subprocess.run(['git', 'clone', '-q', '--depth', '1', url, path], check=True)
    return path


def import_from(repo_dir, module, names):
    """외부 저장소 코드를 불러옴. 저장소마다 utils/models 같은 흔한 이름이 겹쳐서, 불러온 뒤 sys.modules에서 지움"""
    before = set(sys.modules)
    sys.path.insert(0, repo_dir)
    try:
        mod = importlib.import_module(module)
        return [getattr(mod, n) for n in names]
    finally:
        sys.path.remove(repo_dir)
        for k in set(sys.modules) - before:
            if k.split('.')[0] in ('utils', 'models', 'ted'):
                del sys.modules[k]


def run_torch(net, x, mult):
    """크기를 mult의 배수로 맞춰 추론 → 원래 크기로 되돌림"""
    h, w = x.shape[2:]
    H, W = -(-h // mult) * mult, -(-w // mult) * mult
    t = torch.from_numpy(x).to(TORCH_DEV)
    if (H, W) != (h, w):
        t = torch.nn.functional.interpolate(t, (H, W), mode='bilinear', align_corners=False)
    with torch.no_grad():
        out = net(t)
    out = out[-1] if isinstance(out, (list, tuple)) else out
    out = torch.nn.functional.interpolate(out, (h, w), mode='bilinear', align_corners=False)
    return out[0, 0].cpu().numpy().astype(np.float32)


BGR_MEAN = np.array([103.939, 116.779, 123.68], np.float32)


def minmax(p):
    """TEED/DexiNed 원 저자 코드와 같은 출력 정규화 (sigmoid 뒤 이미지별 최소~최대를 0~1로)"""
    return ((p - p.min()) / (p.max() - p.min() + 1e-12)).astype(np.float32)

# PiDiNet: RGB, ImageNet 정규화, 출력 마지막 = fuse(sigmoid 적용됨)
PIDINET_DIR = clone_repo('https://github.com/hellozhuo/pidinet.git', 'pidinet')
(pidinet_builder,) = import_from(PIDINET_DIR, 'models.pidinet', ['pidinet'])
pidinet_net = pidinet_builder(type('Args', (), {'config': 'carv4', 'dil': True, 'sa': True})())   # 논문 기본 설정
_sd = torch.load(os.path.join(PIDINET_DIR, 'trained_models', 'table5_pidinet.pth'), map_location='cpu')['state_dict']
pidinet_net.load_state_dict({k.replace('module.', '', 1): v for k, v in _sd.items()})
pidinet_net.eval().to(TORCH_DEV)


def detect_pidinet(img):
    x = (bgr2rgb(img).astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return run_torch(pidinet_net, x.transpose(2, 0, 1)[None].astype(np.float32), 16)


# TEED: BGR − 평균, 512보다 작으면 1.5배 키워서 추론 (원 저자 테스트 방식), 출력 마지막 = fuse logit
TEED_DIR = clone_repo('https://github.com/xavysp/TEED.git', 'TEED')
(TED,) = import_from(TEED_DIR, 'ted', ['TED'])
teed_net = TED()
teed_net.load_state_dict(torch.load(os.path.join(TEED_DIR, 'checkpoints', 'BIPED', '5', '5_model.pth'), map_location='cpu'))
teed_net.eval().to(TORCH_DEV)


def detect_teed(img):
    h, w = img.shape[:2]
    big = cv2.resize(img, (0, 0), fx=1.5, fy=1.5) if min(h, w) < 512 else img
    x = (big.astype(np.float32) - BGR_MEAN).transpose(2, 0, 1)[None]
    logit = run_torch(teed_net, x, 8)
    return cv2.resize(minmax(1 / (1 + np.exp(-logit))), (w, h), interpolation=cv2.INTER_AREA)


# DexiNed (kornia): BGR − 평균, 출력 = fuse logit
try:
    from kornia.models.dexined import DexiNed
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kornia'], check=True)
    from kornia.models.dexined import DexiNed
dexined_net = DexiNed(pretrained=True).eval().to(TORCH_DEV)


def detect_dexined(img):
    x = (img.astype(np.float32) - BGR_MEAN).transpose(2, 0, 1)[None]
    return minmax(1 / (1 + np.exp(-run_torch(dexined_net, x, 16))))


DETECTORS.update({'pidinet': detect_pidinet, 'teed': detect_teed, 'dexined': detect_dexined})
print('검출기:', list(DETECTORS), '| torch', TORCH_DEV)

if RUN_DEMO:
    img0 = load_image(sample_id)
    probs0 = {k: f(img0) for k, f in DETECTORS.items()}
    show([bgr2rgb(img0)] + list(probs0.values()), ['image'] + list(probs0.keys()), save='1_detectors.png')

# %% [markdown]
# ## 2. 후처리: edge 맵 → 닫힌 구역
#
# 색칠은 **구역 단위**로 채워져요. 선이 한 군데라도 끊기면 이웃 구역과 합쳐져서 색이 새요.
# 그래서 아래 순서로 "항상 닫힌" 구역을 만들어요.
#
# 1. **이진화**: 확률 ≥ `thresh` 인 픽셀을 edge로
# 2. **얇게 만들기**(skeletonize): HED·Sobel처럼 두꺼운 선을 1픽셀로. 구역이 선에 먹히지 않게
# 3. **끊긴 선 잇기**(morphological closing, `close_ksize`): 가까운 끊김을 메움 → 다시 얇게
# 4. **구역 분할**: edge가 아닌 픽셀끼리 연결 요소(4-연결)로 묶음
#    (선은 8-연결, 구역은 4-연결 → 대각선 1픽셀 선 사이로도 새지 않음)
# 5. **작은 구역 병합**: 면적 < `min_area_ratio` × 전체인 구역과 edge 픽셀을 가장 가까운 구역에 붙임
# 6. **경계선**: 최종 구역 지도의 경계에서 선을 다시 그림 → 칠해지는 범위와 선이 항상 일치

# %%
def fill_nearest(lab):
    """lab==0 인 픽셀을 가장 가까운 0이 아닌 구역 번호로 채움 (거리 변환)"""
    mask = lab == 0
    if not mask.any():
        return lab
    if mask.all():
        return np.ones_like(lab)
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return lab[tuple(idx)]


def relabel(region):
    """같은 번호라도 떨어져 있으면 다른 구역으로, 번호는 0부터 연속으로"""
    return sk_label(region + 1, connectivity=1, background=0).astype(np.int32) - 1


def edges_to_regions(prob, thresh=0.5, thin=True, close_ksize=3, min_area_ratio=0.002, return_steps=False):
    steps = {}
    e = prob >= thresh                                                        # 1. 이진화
    steps['1 binary'] = e
    region = binary_to_regions(e, thin, close_ksize, min_area_ratio, steps)
    return (region, steps) if return_steps else region


def binary_to_regions(e, thin=True, close_ksize=3, min_area_ratio=0.002, steps=None):
    """이진 edge → 닫힌 구역 (2~5단계). 후처리 버전들이 공통으로 사용"""
    steps = {} if steps is None else steps
    if thin:
        e = skeletonize(e)                                                    # 2. 얇게
        steps['2 thin'] = e
    if close_ksize > 1:                                                       # 3. 끊긴 선 잇기
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        e = cv2.morphologyEx(e.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        if thin:
            e = skeletonize(e)
        steps['3 closed'] = e

    _, lab = cv2.connectedComponents((~e).astype(np.uint8), connectivity=4)  # 4. 구역 (edge=0)
    steps['4 regions(raw)'] = lab

    min_area = max(20, int(lab.size * min_area_ratio))                        # 5. 작은 구역 병합
    sizes = np.bincount(lab.ravel())
    small = sizes < min_area
    small[0] = True                                                           # edge 픽셀도 채울 대상
    lab = np.where(small[lab], 0, lab)
    region = relabel(fill_nearest(lab))
    steps['5 regions(merged)'] = region
    return region


def region_boundary(region):
    """6. 오른쪽/아래 픽셀과 구역이 다르면 선 (팀 원본 coloringBook.js와 같은 규칙)"""
    line = np.zeros(region.shape, bool)
    line[:, :-1] |= region[:, :-1] != region[:, 1:]
    line[:-1, :] |= region[:-1, :] != region[1:, :]
    return line


def random_colors(region):
    rng = np.random.default_rng(0)
    pal = rng.integers(60, 255, (region.max() + 1, 3), dtype=np.uint8)
    return pal[region]


# 단계별로 눈으로 확인 (HED)
if RUN_DEMO:
    region0, steps0 = edges_to_regions(probs0['hed'], thresh=0.3, return_steps=True)
    vis = [probs0['hed']] + [random_colors(v) if 'regions' in k else v for k, v in steps0.items()]
    vis.append(region_boundary(region0))
    show(vis, ['HED prob'] + list(steps0.keys()) + ['6 line'], cols=4, save='2_postprocess_steps.png')

# %% [markdown]
# ### 2-2. 후처리 v2: NMS + 이중 임계값(hysteresis)
#
# v1의 1~2단계(단순 임계값 → skeletonize)를 Canny의 두 아이디어로 바꿔요. 3단계 이후는 v1과 같아요.
#
# - **NMS (Non-Maximum Suppression)**: 선에 수직인 방향으로 양옆 픽셀보다 작으면 지워서 선 가운데만 남김.
#   선 방향은 edge 맵의 2차 미분(Hessian)으로 구함 (HED/BSDS 평가에서 쓰는 방식과 같은 원리).
#   skeletonize보다 잔가지가 적고, 선이 실제 경계의 가장 강한 위치에 놓여요.
# - **이중 임계값**: 강한 선(≥ `thresh`)에 이어진 약한 선(≥ `thresh × hysteresis_low`)까지 살림.
#   단순 임계값에서 끊기던 약한 경계 구간이 이어질 수 있어요.
#
# 각 아이디어의 효과를 따로 보려고 `v2-nms`(NMS만), `v2-hyst`(이중 임계값만), `v2`(둘 다)로 나눠 등록해요.

# %%
def edge_nms(prob, sigma=2.0, margin=1.01):
    """선에 수직 방향의 양옆 이웃보다 작으면 0으로. 반환: 살아남은 위치만 원래 값"""
    E = prob.astype(np.float32)
    S = cv2.GaussianBlur(E, (0, 0), sigma)
    gy, gx = np.gradient(S)
    hxy, hxx = np.gradient(gx)
    hyy, _ = np.gradient(gy)
    # Hessian 고유벡터: 곡률(2차 미분)의 절댓값이 큰 방향 = 선에 수직인 방향
    # (선 중심은 위로 볼록(−), 바깥 비탈은 아래로 볼록(+)이라 부호가 아니라 절댓값으로 골라야 함)
    theta = 0.5 * np.arctan2(2 * hxy, hxx - hyy)      # 큰 고유값(λ+)의 방향
    half_tr = (hxx + hyy) / 2
    root = np.sqrt(((hxx - hyy) / 2) ** 2 + hxy ** 2)
    normal = np.where(np.abs(half_tr + root) >= np.abs(half_tr - root), theta, theta + np.pi / 2)
    nx, ny = np.cos(normal), np.sin(normal)
    h, w = E.shape
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    nb1 = cv2.remap(E, X + nx.astype(np.float32), Y + ny.astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    nb2 = cv2.remap(E, X - nx.astype(np.float32), Y - ny.astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    keep = (E * margin >= nb1) & (E * margin >= nb2)
    return np.where(keep, E, 0)


def hysteresis(p, low, high):
    """강한 선(≥high)과 8-연결로 이어진 약한 선(≥low)까지 살림"""
    lab, n = ndimage.label(p >= low, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[p >= high])] = True
    keep[0] = False
    return keep[lab]


def edges_to_regions_v2(prob, thresh=0.5, thin=True, close_ksize=3, min_area_ratio=0.002,
                        nms=True, hysteresis_low=0.4, return_steps=False):
    steps = {}
    p = edge_nms(prob) if nms else prob                                   # 1a. NMS
    if nms:
        steps['1a nms'] = p
    if hysteresis_low:                                                    # 1b. 이중 임계값
        e = hysteresis(p, thresh * hysteresis_low, thresh)
    else:
        e = p >= thresh
    steps['1 binary'] = e
    region = binary_to_regions(e, thin, close_ksize, min_area_ratio, steps)   # 2~5단계는 v1과 동일
    return (region, steps) if return_steps else region


# 후처리 버전 등록부: 새 버전을 만들면 이름을 붙여 추가 → dev_eval.py --pp v1 v2 ... 로 비교
POSTPROCESS = {
    'v1':      edges_to_regions,
    'v2-nms':  partial(edges_to_regions_v2, hysteresis_low=0),
    'v2-hyst': partial(edges_to_regions_v2, nms=False),
    'v2':      edges_to_regions_v2,
}

if RUN_DEMO:
    region0b, steps0b = edges_to_regions_v2(probs0['hed'], thresh=0.3, return_steps=True)
    vis = [probs0['hed']] + [random_colors(v) if 'regions' in k else v for k, v in steps0b.items()]
    vis.append(region_boundary(region0b))
    show(vis, ['HED prob'] + list(steps0b.keys()) + ['6 line'], cols=4, save='2b_postprocess_v2_steps.png')

# %% [markdown]
# ### 비교 기준: 팀 원본 방식 (k-means 색 구역) Python 포팅
#
# `web/src/coloring/coloringBook.js`의 로직을 옮겼어요 (원본 파일은 수정하지 않음).
# 블러 → k-means(K=7) → 3×3 다수결 ×2 → 작은 구역 병합 → 연결 요소.
# 작은 구역 병합은 원본의 "이웃 색 투표" 대신 "가장 가까운 구역"으로 근사했어요.

# %%
def kmeans_quantize(img, K=7, n_sample=5000):
    """픽셀 샘플로 k-means 학습 → 전체 픽셀 할당. 반환: labels(H,W), centers(K,3) BGR"""
    px = img.reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(SEED)
    sample = px[rng.choice(len(px), min(n_sample, len(px)), replace=False)]
    cv2.setRNGSeed(SEED)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centers = cv2.kmeans(sample, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    d = ((px[:, None, :] - centers[None]) ** 2).sum(-1)       # 전체 픽셀 → 가장 가까운 중심
    return d.argmin(1).reshape(img.shape[:2]), centers


def mode_filter(labels, K):
    counts = np.stack([cv2.boxFilter((labels == k).astype(np.float32), -1, (3, 3)) for k in range(K)])
    return counts.argmax(0)


def regions_kmeans_baseline(img, K=7, min_area_ratio=0.002):
    blurred = cv2.blur(cv2.blur(img, (3, 3)), (3, 3))
    labels, _ = kmeans_quantize(blurred, K)
    labels = mode_filter(mode_filter(labels, K), K)
    comp = relabel(labels) + 1
    min_area = max(40, int(comp.size * min_area_ratio))
    sizes = np.bincount(comp.ravel())
    comp = np.where(sizes[comp] < min_area, 0, comp)
    return relabel(fill_nearest(comp))

# %% [markdown]
# ## 3. 색칠공부판 + 팔레트
#
# - **팔레트**: 원본 사진 색을 k-means로 K개로 묶은 대표색 (많이 쓰인 순)
# - **색칠판**: 흰 바탕 + 경계선
# - **정답 예시**: 구역마다 평균색과 가장 가까운 팔레트 색으로 칠한 결과

# %%
def build_coloring_book(img, region, K=7):
    blurred = cv2.blur(img, (5, 5))
    labels, centers = kmeans_quantize(blurred, K)
    order = np.argsort(-np.bincount(labels.ravel(), minlength=K))
    palette = centers[order][:, ::-1].round().astype(np.uint8)          # BGR → RGB

    line = region_boundary(region)
    board = np.full(img.shape, 255, np.uint8)
    board[line] = 35

    n = region.max() + 1
    cnt = np.bincount(region.ravel(), minlength=n)[:, None]
    rgb = bgr2rgb(img).reshape(-1, 3).astype(np.float64)
    means = np.stack([np.bincount(region.ravel(), rgb[:, c], minlength=n) for c in range(3)], 1) / np.maximum(cnt, 1)
    nearest = ((means[:, None, :] - palette[None].astype(np.float64)) ** 2).sum(-1).argmin(1)
    colored = palette[nearest][region]
    colored[line] = 35
    return dict(region=region, line=line, board=board, palette=palette, colored=colored, n_regions=n)


def palette_strip(palette, h=40):
    return np.repeat(np.repeat(palette[None], h, 0), h, 1)


# 방법별 기본 후처리 파라미터 (5장 실험으로 조정해 보세요)
PP = {
    'canny': dict(thresh=0.5, thin=True, close_ksize=3, min_area_ratio=0.002),
    'sobel': dict(thresh=0.35, thin=True, close_ksize=3, min_area_ratio=0.002),
    'hed':   dict(thresh=0.3, thin=True, close_ksize=3, min_area_ratio=0.002),
}
# 추가 딥러닝 모델은 튜닝 없이 HED와 같은 기본값 사용 (공정한 "기본 모델 + 기존 후처리" 비교)
for _m in ('pidinet', 'teed', 'dexined'):
    PP[_m] = dict(PP['hed'])

METHODS = {'kmeans (team)': regions_kmeans_baseline}
for _m in ('canny', 'sobel', 'hed', 'pidinet', 'teed', 'dexined'):
    METHODS[_m] = partial(lambda img, m: edges_to_regions(DETECTORS[m](img), **PP[m]), m=_m)


def compare_methods(img, save=None):
    ims, titles = [], []
    for name, fn in METHODS.items():
        book = build_coloring_book(img, fn(img))
        ims += [book['board'], book['colored']]
        titles += [f'{name}: board ({book["n_regions"]} regions)', f'{name}: colored']
    show(ims, titles, cols=4, save=save)
    return book


if RUN_DEMO:
    book0 = compare_methods(img0, save='3_compare_methods.png')
    show([palette_strip(book0['palette'])], ['palette'], size=6)

# %% [markdown]
# ## 4. 평가
#
# BSDS500 테스트 이미지에서 사람 주석(GT)과 비교해요. 주석자가 여러 명이라 각 GT와 비교한 뒤 평균을 내요.
#
# 기호: N = 전체 픽셀 수, Rᵢ = 내 구역, Gⱼ = 정답 구역
#
# | 지표 | 정의 | 의미 | 좋은 방향 |
# |---|---|---|---|
# | **Leakage** | 1 − (1/N) Σᵢ maxⱼ \|Rᵢ ∩ Gⱼ\| | 내 구역 안에 다른 정답 구역이 섞인 비율 = **칠하면 색이 새는 정도** (1 − Purity) | ↓ |
# | **Fragmentation** | 1 − (1/N) Σⱼ maxᵢ \|Gⱼ ∩ Rᵢ\| | 정답 구역이 내 구역 여러 개로 갈라진 비율 = 칠하기 번거로움 (1 − Inverse Purity) | ↓ |
# | **Region F** (주 지표) | (1−Leakage)와 (1−Fragmentation)의 조화평균 | 색칠공부판으로서의 종합 품질 | ↑ |
# | **Boundary F** (보조 지표) | 선 Precision·Recall의 조화평균 (허용 거리 = 대각선 × 0.0075) | 선이 정답 경계 위치에 그려졌나 | ↑ |
# | regions / ms | | 구역 수, 처리 시간 (웹 실시간 데모 가능성) | |
#
# Leakage와 Fragmentation은 서로 반대 방향이라 (구역을 잘게 나누면 Leakage↓ Fragmentation↑) 하나로 합친 Region F를 주 지표로 써요.
#
# **채택 규칙 (실험 전에 고정)**: dev set에서 Region F가 오르고, Boundary F가 0.01 넘게 떨어지지 않으면 채택.
#
# ※ Boundary F는 공식 BSDS 벤치마크(MATLAB, 1:1 매칭)를 단순화한 근사치예요.

# %%
def boundary_f(line, gt_bnds, tol_ratio=0.0075):
    tol = tol_ratio * np.hypot(*line.shape)
    if not line.any():
        return 0.0, 0.0, 0.0
    d_gt = ndimage.distance_transform_edt(~np.any(gt_bnds, 0))
    P = float((d_gt[line] <= tol).mean())
    d_pred = ndimage.distance_transform_edt(~line)
    R = float(np.mean([(d_pred[g] <= tol).mean() for g in gt_bnds if g.any()]))
    F = 2 * P * R / (P + R) if P + R > 0 else 0.0
    return P, R, F


def purity(a, b):
    """a의 각 구역이 b의 한 구역에 얼마나 몰려 있나 (1이면 완전히 b 안에 들어감)"""
    na, nb = a.max() + 1, b.max() + 1
    cont = np.bincount((a.astype(np.int64) * nb + b).ravel(), minlength=na * nb).reshape(na, nb)
    return cont.max(1).sum() / a.size


def evaluate_region(region, gt_segs, gt_bnds):
    P, R, F = boundary_f(region_boundary(region), gt_bnds)
    leak = 1 - np.mean([purity(region, s) for s in gt_segs])
    frag = 1 - np.mean([purity(s, region) for s in gt_segs])
    a, b = 1 - leak, 1 - frag
    region_F = 2 * a * b / (a + b) if a + b > 0 else 0.0    # 주 지표: 안 새는 정도와 안 쪼개진 정도의 조화평균
    return dict(region_F=region_F, F=F, P=P, R=R, leakage=leak, fragmentation=frag, regions=int(region.max() + 1))


def run_eval(methods, ids):
    rows = []
    for i, img_id in enumerate(ids):
        img = load_image(img_id)
        segs, bnds = load_gt(img_id)
        for name, fn in methods.items():
            t0 = time.perf_counter()
            region = fn(img)
            ms = (time.perf_counter() - t0) * 1000
            rows.append(dict(id=img_id, method=name, ms=ms, **evaluate_region(region, segs, bnds)))
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(ids)}')
    return pd.DataFrame(rows)

# %%
# ── 4-2. 평가 실행 (test) ────────────────────────────────
# ※ 개선 실험은 dev set(val 20장, edge/dev_eval.py)으로 하고, test는 최종 결과 확인용
eval_ids = test_ids[:N_EVAL]
print(f'평가: {len(eval_ids)}장 × {len(METHODS)}개 방법')
df = run_eval(METHODS, eval_ids)
summary = df.groupby('method')[['region_F', 'F', 'P', 'R', 'leakage', 'fragmentation', 'regions', 'ms']].mean()
summary = summary.sort_values('region_F', ascending=False).round(3)
summary.to_csv(os.path.join(OUT, 'eval_summary.csv'))
print(summary.to_string())

# %%
fig, ax = plt.subplots(figsize=(6, 4.5))
for name, g in df.groupby('method'):
    ax.scatter(g['leakage'], g['fragmentation'], label=name, alpha=0.6, s=25)
ax.set_xlabel('Leakage (lower is better)')
ax.set_ylabel('Fragmentation (lower is better)')
ax.set_title('Leakage vs Fragmentation (1 dot = 1 image)')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT, '4_leak_vs_frag.png'), dpi=90)
plt.show()
plt.close(fig)

# %% [markdown]
# ## 5. 후처리 파라미터 실험 (ablation)
#
# edge 확률 맵은 이미지당 한 번만 계산해 두고, 임계값(`thresh`)과 선 잇기 커널(`close_ksize`)만 바꿔서 비교해요.
#
# - HED: 임계값이 낮을수록 선이 많아져 Leakage↓, 대신 Fragmentation↑
# - Canny: 선이 이미 이진이라 임계값 의미 없음 → `close_ksize`(끊김 메우기)가 Leakage를 얼마나 줄이는지가 관건

# %%
N_SWEEP = min(10, len(eval_ids))
SWEEP_GRID = {
    'hed':   dict(thresh=[0.2, 0.3, 0.4, 0.5], close_ksize=[1, 3, 5, 7]),
    'canny': dict(thresh=[0.5],                close_ksize=[1, 3, 5, 7, 9, 11]),
}
gt_cache = [(load_image(i), *load_gt(i)) for i in eval_ids[:N_SWEEP]]

sweeps = {}
for det, grid in SWEEP_GRID.items():
    probs = [DETECTORS[det](img) for img, _, _ in gt_cache]
    rows = []
    for thresh in grid['thresh']:
        for ksize in grid['close_ksize']:
            for prob, (_, segs, bnds) in zip(probs, gt_cache):
                region = edges_to_regions(prob, thresh=thresh, close_ksize=ksize)
                rows.append(dict(thresh=thresh, close_ksize=ksize, **evaluate_region(region, segs, bnds)))
    sweeps[det] = pd.DataFrame(rows).groupby(['thresh', 'close_ksize']).mean().round(3)
    sweeps[det].to_csv(os.path.join(OUT, f'sweep_{det}.csv'))
    print(f'\n[{det}]')
    print(sweeps[det][['region_F', 'F', 'leakage', 'fragmentation', 'regions']].to_string())

# %% [markdown]
# ## 6. 내 사진 / 웹캠으로 해보기
#
# Colab에서는 사진을 업로드하거나 웹캠으로 찍을 수 있어요. 웹 앱과 같은 크기(가로 480)로 줄여서 처리해요.

# %%
def run_on_image(img, width=480, save=None):
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    return compare_methods(img, save=save)


if IN_COLAB:
    from google.colab import files
    uploaded = files.upload()          # 업로드 창이 뜨면 사진 선택 (취소하면 건너뜀)
    for name, data in uploaded.items():
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        run_on_image(img, save=f'6_{os.path.splitext(name)[0]}.png')
else:
    print('로컬 실행: run_on_image(cv2.imread("사진경로")) 로 직접 해보세요')

# %%
# (선택) Colab 웹캠으로 한 장 찍기
WEBCAM_JS = '''
async function takePhoto(quality) {
  const div = document.createElement('div');
  const btn = document.createElement('button'); btn.textContent = '찍기';
  const video = document.createElement('video'); video.style.display = 'block';
  const stream = await navigator.mediaDevices.getUserMedia({video: true});
  div.appendChild(video); div.appendChild(btn); document.body.appendChild(div);
  video.srcObject = stream; await video.play();
  google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
  await new Promise(resolve => btn.onclick = resolve);
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  stream.getVideoTracks()[0].stop(); div.remove();
  return canvas.toDataURL('image/jpeg', quality);
}
'''

def webcam_photo():
    import base64
    from IPython.display import Javascript, display
    from google.colab.output import eval_js
    display(Javascript(WEBCAM_JS))
    data = eval_js('takePhoto(0.9)')
    buf = np.frombuffer(base64.b64decode(data.split(',')[1]), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

# 쓰려면 아래 줄 주석 해제
# if IN_COLAB: run_on_image(webcam_photo(), save='6_webcam.png')
