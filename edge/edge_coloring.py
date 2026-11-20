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

WORK   = '/content/edge_work' if IN_COLAB else os.environ.get('EDGE_WORK', os.path.abspath('edge_work'))
DATA   = os.path.join(WORK, 'data')
MODELS = os.path.join(WORK, 'models')
OUT    = os.path.join(WORK, 'outputs')
for d in (DATA, MODELS, OUT):
    os.makedirs(d, exist_ok=True)

N_EVAL = int(os.environ.get('N_EVAL', 30))   # 평가에 쓸 테스트 이미지 수 (최대 200)
SEED = 0
np.random.seed(SEED)
print('Colab' if IN_COLAB else 'Local', '| 작업 폴더:', WORK, '| OpenCV', cv2.__version__)


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

IMG_DIR = os.path.join(BSDS_DIR, 'BSDS500', 'data', 'images', 'test')
GT_DIR  = os.path.join(BSDS_DIR, 'BSDS500', 'data', 'groundTruth', 'test')
test_ids = sorted(f[:-4] for f in os.listdir(IMG_DIR) if f.endswith('.jpg'))
print('테스트 이미지:', len(test_ids), '장')


def load_image(img_id):
    return cv2.imread(os.path.join(IMG_DIR, img_id + '.jpg'))   # BGR


def load_gt(img_id):
    """사람 주석자(보통 5명)마다 구역 분할(Segmentation)과 경계(Boundaries) 하나씩"""
    gts = sio.loadmat(os.path.join(GT_DIR, img_id + '.mat'))['groundTruth'][0]
    segs = [g['Segmentation'][0, 0].astype(np.int32) for g in gts]
    bnds = [g['Boundaries'][0, 0].astype(bool) for g in gts]
    return segs, bnds


sample_id = test_ids[0]
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
    return (region, steps) if return_steps else region


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
region0, steps0 = edges_to_regions(probs0['hed'], thresh=0.3, return_steps=True)
vis = [probs0['hed']] + [random_colors(v) if 'regions' in k else v for k, v in steps0.items()]
vis.append(region_boundary(region0))
show(vis, ['HED prob'] + list(steps0.keys()) + ['6 line'], cols=4, save='2_postprocess_steps.png')

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
METHODS = {
    'kmeans (team)': regions_kmeans_baseline,
    'canny': lambda img: edges_to_regions(detect_canny(img), **PP['canny']),
    'sobel': lambda img: edges_to_regions(detect_sobel(img), **PP['sobel']),
    'hed':   lambda img: edges_to_regions(detect_hed(img), **PP['hed']),
}


def compare_methods(img, save=None):
    ims, titles = [], []
    for name, fn in METHODS.items():
        book = build_coloring_book(img, fn(img))
        ims += [book['board'], book['colored']]
        titles += [f'{name}: board ({book["n_regions"]} regions)', f'{name}: colored']
    show(ims, titles, cols=4, save=save)
    return book


book0 = compare_methods(img0, save='3_compare_methods.png')
show([palette_strip(book0['palette'])], ['palette'], size=6)

# %% [markdown]
# ## 4. 평가
#
# BSDS500 테스트 이미지에서 사람 주석(GT)과 비교해요. 주석자가 여러 명이라 각 GT와 비교한 뒤 평균을 내요.
#
# | 지표 | 의미 | 좋은 방향 |
# |---|---|---|
# | **Boundary F** | 경계선이 GT 경계와 얼마나 겹치나 (허용 거리 = 대각선 × 0.0075) | ↑ |
# | **Leakage** | 내 구역 하나에 서로 다른 GT 구역이 섞인 비율 = **칠하면 색이 새는 정도** | ↓ |
# | **Fragmentation** | GT 구역 하나가 내 구역 여러 개로 쪼개진 정도 = 칠하기 번거로움 | ↓ |
# | regions / ms | 구역 수, 처리 시간 (웹 실시간 데모 가능성) | |
#
# Leakage와 Fragmentation은 서로 반대 방향이라 (구역을 잘게 나누면 Leakage↓ Fragmentation↑) 둘을 같이 봐야 해요.
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
    return dict(P=P, R=R, F=F, leakage=leak, fragmentation=frag, regions=int(region.max() + 1))


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


eval_ids = test_ids[:N_EVAL]
print(f'평가: {len(eval_ids)}장 × {len(METHODS)}개 방법')
df = run_eval(METHODS, eval_ids)
summary = df.groupby('method')[['F', 'P', 'R', 'leakage', 'fragmentation', 'regions', 'ms']].mean()
summary = summary.sort_values('F', ascending=False).round(3)
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
    print(sweeps[det][['F', 'leakage', 'fragmentation', 'regions']].to_string())

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
