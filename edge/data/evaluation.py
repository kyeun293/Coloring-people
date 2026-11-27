"""
evaluation.py: BSDS500 표준 평가 (MATLAB 공식 벤치마크를 Python으로 옮김, 담당: Taein)

A (edge detection): ODS F, OIS F, AP           ← BSDS boundary benchmark (Martin et al. 2004)
B (segmentation):  SC, PRI, VI, F_b            ← BSDS region benchmark (Arbeláez et al. 2011)
                   + 보조 지표 Leakage, Fragmentation (색칠공부용, 비표준)

사용 (A):
    from evaluation import EdgeEvaluator
    ev = EdgeEvaluator()                       # nthresh=99, max_dist=0.0075, apply_nms=True
    for b in DataLoader(EdgeDataset('val', ('bsds',)), batch_size=1):
        prob = torch.sigmoid(model(b['image']))[0, 0].cpu().numpy()     # (H,W) 0~1
        ev.add(prob, b['bnd'][0].numpy(), img_id=b['id'][0])
    print(ev.summary())                        # {'ODS': .., 'OIS': .., 'AP': .., ...}

사용 (B):
    from evaluation import SegEvaluator
    ev = SegEvaluator()
    for s in iter_eval('val', 'bsds'):
        region = my_segmentation(s['image'])   # (H,W) int 구역 번호
        ev.add(region, s['seg'], s['bnd'], img_id=s['id'])
    print(ev.summary())                        # {'SC': .., 'PRI': .., 'VI': .., 'F_b': .., ...}

MATLAB 원본과 같게 맞춘 부분:
- 임계값: linspace(1/(n+1), 1-1/(n+1), n), 예측 맵을 임계값으로 자른 뒤 thin(무한 반복)으로 1픽셀화
- 매칭: correspondPixels와 같은 결과 (허용 거리 안에서 1:1 최대 매칭. 원본은 최소 비용 매칭이지만
  짝 없는 픽셀 비용이 매우 커서 매칭 개수는 최대 매칭과 같음). 허용 거리 = max_dist × 이미지 대각선
- 주석자 여러 명: 예측 픽셀은 누구와든 매칭되면 정답(P), recall은 모든 주석자의 GT 픽셀을 합쳐서(R)
- ODS: 임계값마다 전체 이미지의 개수를 합산해 P, R → 인접 임계값 사이를 100단계 보간해 최대 F
- OIS: 이미지마다 F가 최대인 임계값의 개수를 합산해 P, R → F
- AP: recall 0, 0.01, …, 1에서 precision을 선형 보간해 합한 뒤 × 0.01 (곡선 밖은 0)
- NMS: Dollár edges 툴박스의 edgesNms (HED, RCF, PiDiNet 평가 스크립트와 같은 설정: r=1, s=5, m=1.01)
- SC: 주석자 전체의 (면적 × 최대 IoU) 합 ÷ 면적 합. 데이터셋은 개수를 합산 (픽셀 가중)
- PRI: 주석자별 Rand Index의 평균 (= p_ij 정의와 수학적으로 같음). 데이터셋은 이미지 평균
- VI: 자연로그, 주석자별 평균. 데이터셋은 이미지 평균
"""
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
from skimage.morphology import thin

MAX_DIST = 0.0075


# ── NMS (Dollár edgesNms) ───────────────────────────────
def _conv_tri(E, r):
    """Dollár convTri: 삼각형 필터 (반지름 r), 대칭 패딩"""
    f = np.concatenate([np.arange(1, r + 2), np.arange(r, 0, -1)]).astype(np.float64)
    f /= f.sum()
    E = ndimage.convolve1d(E, f, axis=0, mode='reflect')
    return ndimage.convolve1d(E, f, axis=1, mode='reflect')


def _interp(E, x, y):
    h, w = E.shape
    x = np.clip(x, 0, w - 1.001); y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - x0, y - y0
    return (E[y0, x0] * (1 - dx) * (1 - dy) + E[y0, x0 + 1] * dx * (1 - dy)
            + E[y0 + 1, x0] * (1 - dx) * dy + E[y0 + 1, x0 + 1] * dx * dy)


def edge_nms(E, r=1, s=5, m=1.01):
    """HED/RCF/PiDiNet 평가 스크립트와 같은 NMS:
    O = mod(atan(Oyy·sign(-Oxy) / (Oxx+1e-5)), π)  (convTri(E,4)의 2차 미분으로 구한 방향)
    → 그 방향 ±1 픽셀 값보다 작으면(×1.01) 0 → 테두리 s픽셀은 거리에 비례해 약하게"""
    E = np.asarray(E, np.float64)
    Oy, Ox = np.gradient(_conv_tri(E, 4))
    _, Oxx = np.gradient(Ox)
    Oyy, Oxy = np.gradient(Oy)
    O = np.mod(np.arctan(Oyy * np.sign(-Oxy) / (Oxx + 1e-5)), np.pi)
    h, w = E.shape
    ys, xs = np.nonzero(E)
    e = E[ys, xs] * m
    c, sn = np.cos(O[ys, xs]), np.sin(O[ys, xs])
    keep = np.ones(len(ys), bool)
    for d in range(-r, r + 1):
        if d:
            keep &= e >= _interp(E, xs + d * c, ys + d * sn)
    out = np.zeros_like(E)
    out[ys[keep], xs[keep]] = E[ys[keep], xs[keep]]
    s = min(s, w // 2, h // 2)
    for i in range(s):
        out[:, i] *= i / s; out[:, w - 1 - i] *= i / s
        out[i, :] *= i / s; out[h - 1 - i, :] *= i / s
    return out.astype(np.float32)


# ── 픽셀 매칭 (correspondPixels) ────────────────────────
# scipy의 maximum_bipartite_matching은 경계선처럼 긴 사슬 모양 그래프에서 가끔 수십 초씩 멈춰서
# (방향에 따라 0.2초 ↔ 70초), 최악의 경우에도 빠른 Hopcroft-Karp를 numba로 직접 구현
try:
    from numba import njit
except ImportError:          # numba가 없으면 scipy로 (결과는 같고 가끔 느릴 수 있음)
    njit = None

if njit is not None:
    @njit(cache=True)
    def _hopcroft_karp(indptr, indices, n_left, n_right):
        INF = 1 << 30
        match_l = np.full(n_left, -1, np.int64)
        match_r = np.full(n_right, -1, np.int64)
        for u in range(n_left):                                   # 탐욕 초기 매칭
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                if match_r[v] == -1:
                    match_l[u] = v
                    match_r[v] = u
                    break
        dist = np.empty(n_left, np.int64)
        queue = np.empty(n_left, np.int64)
        it = np.empty(n_left, np.int64)
        stack_u = np.empty(n_left + 1, np.int64)
        stack_v = np.empty(n_left + 1, np.int64)
        while True:
            qh = qt = 0                                           # BFS: 짝 없는 왼쪽 정점에서 층 나누기
            for u in range(n_left):
                if match_l[u] == -1:
                    dist[u] = 0
                    queue[qt] = u
                    qt += 1
                else:
                    dist[u] = INF
            found = False
            while qh < qt:
                u = queue[qh]
                qh += 1
                for k in range(indptr[u], indptr[u + 1]):
                    w = match_r[indices[k]]
                    if w == -1:
                        found = True
                    elif dist[w] == INF:
                        dist[w] = dist[u] + 1
                        queue[qt] = w
                        qt += 1
            if not found:
                break
            for u in range(n_left):
                it[u] = indptr[u]
            for s in range(n_left):                               # DFS: 층을 따라 증가 경로 찾기
                if match_l[s] != -1:
                    continue
                top = 0
                stack_u[0] = s
                while top >= 0:
                    u = stack_u[top]
                    advanced = False
                    while it[u] < indptr[u + 1]:
                        v = indices[it[u]]
                        it[u] += 1
                        w = match_r[v]
                        if w == -1:                               # 증가 경로 발견 → 뒤집기
                            stack_v[top] = v
                            for lv in range(top, -1, -1):
                                uu, vv = stack_u[lv], stack_v[lv]
                                match_l[uu] = vv
                                match_r[vv] = uu
                            top = -1
                            advanced = True
                            break
                        if dist[w] == dist[u] + 1:
                            stack_v[top] = v
                            top += 1
                            stack_u[top] = w
                            advanced = True
                            break
                    if not advanced:
                        dist[u] = INF
                        top -= 1
        return match_l


def _max_matching(G):
    """G: (예측 픽셀 × GT 픽셀) csr → 예측 픽셀마다 짝지은 GT 번호 (-1 = 없음)"""
    if njit is not None:
        return _hopcroft_karp(G.indptr.astype(np.int64), G.indices.astype(np.int64), G.shape[0], G.shape[1])
    return maximum_bipartite_matching(G, perm_type='column')


class _Matcher:
    """후보 예측 픽셀 전체 × 주석자별 GT 픽셀의 '허용 거리 안' 연결 그래프를 한 번만 만들어 두고,
    임계값마다 그 부분집합(행)만 골라 1:1 최대 매칭 (= correspondPixels의 매칭 개수)"""

    def __init__(self, cand_mask, gt_bnds, max_dist):
        self.index = np.full(cand_mask.shape, -1, np.int64)
        yx = np.argwhere(cand_mask)
        self.index[yx[:, 0], yx[:, 1]] = np.arange(len(yx))
        r = max_dist * np.hypot(*cand_mask.shape)
        tree = cKDTree(yx) if len(yx) else None
        self.graphs = []
        for g in gt_bnds:
            gyx = np.argwhere(g)
            if tree is None or len(gyx) == 0:
                self.graphs.append((None, len(gyx)))
                continue
            # ndarray 출력: 거리 0(정확히 겹침)인 쌍도 빠지지 않음
            pairs = tree.sparse_distance_matrix(cKDTree(gyx), r, output_type='ndarray')
            G = csr_matrix((np.ones(len(pairs), np.int8), (pairs['i'], pairs['j'])), shape=(len(yx), len(gyx)))
            self.graphs.append((G, len(gyx)))

    def counts(self, bmap):
        """이진 경계(후보 픽셀의 부분집합) → (cntR, sumR, cntP, sumP)"""
        rows = self.index[bmap]
        assert (rows >= 0).all(), '후보 밖의 픽셀이 있음'
        acc = np.zeros(len(rows), bool)
        cntR = sumR = 0
        for G, n_gt in self.graphs:
            sumR += n_gt
            if G is None or len(rows) == 0:
                continue
            sub = G[rows]                                             # (이번 임계값의 예측 픽셀, GT 픽셀)
            if sub.nnz == 0:
                continue
            m = _max_matching(sub) >= 0
            acc |= m
            cntR += int(m.sum())
        return cntR, sumR, int(acc.sum()), len(rows)


try:   # skimage thin(= MATLAB bwmorph 'thin')과 같은 규칙표를 써서, 전체 이미지 대신 선 픽셀만 계산 (결과 동일, 수십 배 빠름)
    from skimage.morphology._skeletonize import G123_LUT, G123P_LUT
    _NBR = [(-1, -1, 8), (-1, 0, 4), (-1, 1, 2), (0, -1, 16), (0, 1, 1), (1, -1, 32), (1, 0, 64), (1, 1, 128)]
except ImportError:
    G123_LUT = None


def _thin(b):
    if not b.any():
        return b
    if G123_LUT is None:
        return thin(b)
    h, w = b.shape
    skel = np.zeros((h + 2, w + 2), np.uint8)
    skel[1:-1, 1:-1] = b
    ys, xs = np.nonzero(skel)
    while True:
        n_old = len(ys)
        for lut in (G123_LUT, G123P_LUT):
            code = np.zeros(len(ys), np.int32)
            for dy, dx, v in _NBR:
                code += skel[ys + dy, xs + dx].astype(np.int32) * v
            rm = lut[code].astype(bool)
            skel[ys[rm], xs[rm]] = 0
            ys, xs = ys[~rm], xs[~rm]
        if len(ys) == n_old:
            return skel[1:-1, 1:-1].astype(bool)


def boundary_counts(bmap, gt_bnds, max_dist=MAX_DIST):
    """이진 경계 하나 vs 주석자별 GT → (cntR, sumR, cntP, sumP)"""
    return _Matcher(bmap, gt_bnds, max_dist).counts(bmap)


def _f(p, r):
    return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)


def _pr(cntR, sumR, cntP, sumP):
    """MATLAB 원본과 같음: R = cntR/(sumR+(sumR==0)), P = cntP/(sumP+(sumP==0)) → 예측이 없으면 P=0"""
    R = cntR / np.maximum(sumR, 1)
    P = cntP / np.maximum(sumP, 1)
    return P, R


def _interp_best(T, R, P):
    """BSDS maxF: 인접 임계값 사이를 100단계로 보간해 F 최대 지점"""
    best = (T[0], R[0], P[0], _f(P[0], R[0]))
    d = np.linspace(0, 1, 100)
    for i in range(1, len(T)):
        Rt = R[i] * d + R[i - 1] * (1 - d)
        Pt = P[i] * d + P[i - 1] * (1 - d)
        Tt = T[i] * d + T[i - 1] * (1 - d)
        Ft = _f(Pt, Rt)
        k = int(np.argmax(Ft))
        if Ft[k] > best[3]:
            best = (Tt[k], Rt[k], Pt[k], Ft[k])
    return best


# ── A: edge detection 평가 ─────────────────────────────
class EdgeEvaluator:
    """BSDS boundary benchmark. add()로 이미지마다 넣고 summary()로 ODS/OIS/AP"""

    def __init__(self, nthresh=99, max_dist=MAX_DIST, apply_nms=True):
        self.T = np.linspace(1 / (nthresh + 1), 1 - 1 / (nthresh + 1), nthresh)
        self.max_dist, self.apply_nms = max_dist, apply_nms
        self.counts, self.ids = [], []

    def add(self, prob, gt_bnds, img_id=None):
        """prob: (H,W) 0~1 edge 확률 / gt_bnds: (K,H,W) bool 주석자별 경계"""
        prob = np.asarray(prob, np.float32)
        gt_bnds = np.asarray(gt_bnds, bool)
        if gt_bnds.ndim == 2:
            gt_bnds = gt_bnds[None]
        assert prob.shape == gt_bnds.shape[1:], (prob.shape, gt_bnds.shape)
        if self.apply_nms:
            prob = edge_nms(prob)
        matcher = _Matcher(prob >= self.T[0], gt_bnds, self.max_dist)   # 가장 낮은 임계값 = 모든 후보 픽셀
        c = np.zeros((len(self.T), 4), np.int64)
        prev, prev_c = None, None
        for i, t in enumerate(self.T):
            b = _thin(prob >= t)
            if prev is not None and np.array_equal(b, prev):
                c[i] = prev_c                                           # 같은 경계면 다시 매칭하지 않음
            else:
                c[i] = prev_c = matcher.counts(b)
                prev = b
        self.counts.append(c)
        self.ids.append(img_id)
        return c

    def summary(self):
        C = np.stack(self.counts)                                   # (이미지, 임계값, 4)
        tot = C.sum(0)
        P, R = _pr(*tot.T)
        T_ods, R_ods, P_ods, F_ods = _interp_best(self.T, R, P)
        Pi, Ri = _pr(*np.moveaxis(C, 2, 0))                         # (이미지, 임계값)
        best = np.argmax(_f(Pi, Ri), 1)
        oc = C[np.arange(len(C)), best].sum(0)
        P_ois, R_ois = _pr(*oc)
        Ru, idx = np.unique(R[::-1], return_index=True)              # 같은 R이 여러 개면 마지막 것 (MATLAB unique)
        idx = len(R) - 1 - idx
        ap = 0.0
        if len(Ru) > 1:
            Pi_ = np.interp(np.linspace(0, 1, 101), Ru, P[idx], left=np.nan, right=np.nan)
            ap = float(np.nansum(Pi_) * 0.01)
        return dict(ODS=float(F_ods), OIS=float(_f(P_ois, R_ois)), AP=ap,
                    ODS_P=float(P_ods), ODS_R=float(R_ods), ODS_thresh=float(T_ods), n=len(C))

    def per_image(self):
        """이미지별 최적 F (OIS 기준) 목록"""
        out = []
        for img_id, c in zip(self.ids, self.counts):
            P, R = _pr(*c.T)
            F = _f(P, R)
            k = int(np.argmax(F))
            out.append(dict(id=img_id, F=float(F[k]), P=float(P[k]), R=float(R[k]), thresh=float(self.T[k])))
        return out

    def pr_curve(self):
        P, R = _pr(*np.stack(self.counts).sum(0).T)
        return dict(thresh=self.T, P=P, R=R, F=_f(P, R))


# ── B: segmentation 평가 ───────────────────────────────
def _contingency(a, b):
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    a, b = a.ravel(), b.ravel()
    n = np.zeros((a.max() + 1, b.max() + 1), np.float64)
    np.add.at(n, (a, b), 1)
    return n


def rand_index(seg, gt):
    """BSDS compare_segmentations와 같은 식 (픽셀 쌍 N(N-1)/2 기준)"""
    n = _contingency(seg, gt)
    N = n.sum()
    return 1 - ((n.sum(1) ** 2).sum() / 2 + (n.sum(0) ** 2).sum() / 2 - (n ** 2).sum()) / (N * (N - 1) / 2)


def variation_of_information(seg, gt):
    """VI = H(S) + H(G) - 2 I(S;G), 자연로그"""
    n = _contingency(seg, gt)
    p = n / n.sum()
    pa, pb = p.sum(1), p.sum(0)
    h = lambda q: -np.sum(q[q > 0] * np.log(q[q > 0]))
    nz = p > 0
    mi = np.sum(p[nz] * np.log(p[nz] / np.outer(pa, pb)[nz]))
    return h(pa) + h(pb) - 2 * mi


def covering_counts(seg, gt):
    """GT 구역마다 (면적 × 가장 잘 겹치는 예측 구역과의 IoU) 합, 면적 합"""
    n = _contingency(gt, seg)                     # (GT 구역, 예측 구역)
    area_g, area_s = n.sum(1, keepdims=True), n.sum(0, keepdims=True)
    iou = n / (area_g + area_s - n)
    return float((area_g[:, 0] * iou.max(1)).sum()), float(area_g.sum())


def purity(a, b):
    """a의 각 구역이 b의 한 구역에 얼마나 속하는지: (1/N) Σ_i max_j |a_i ∩ b_j|"""
    n = _contingency(a, b)
    return n.max(1).sum() / n.sum()


def seg_to_bdry(seg):
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from unified_data import seg2bdry
    return seg2bdry(seg)


class SegEvaluator:
    """BSDS region benchmark (고정 파라미터 결과 1개 기준) + 구역 경계 F_b + 색칠공부 보조 지표"""

    def __init__(self, max_dist=MAX_DIST):
        self.max_dist = max_dist
        self.rows = []

    def add(self, region, gt_segs, gt_bnds, img_id=None):
        """region: (H,W) int 예측 구역 / gt_segs: (K,H,W) int / gt_bnds: (K,H,W) bool"""
        region = np.asarray(region)
        gt_segs, gt_bnds = np.asarray(gt_segs), np.asarray(gt_bnds, bool)
        if gt_segs.ndim == 2:
            gt_segs, gt_bnds = gt_segs[None], gt_bnds[None]
        assert region.shape == gt_segs.shape[1:], (region.shape, gt_segs.shape)
        cov = np.array([covering_counts(region, g) for g in gt_segs]).sum(0)
        cntR, sumR, cntP, sumP = boundary_counts(seg_to_bdry(region), gt_bnds, self.max_dist)
        row = dict(id=img_id,
                   SC=cov[0] / cov[1], _cov=cov[0], _area=cov[1],
                   PRI=float(np.mean([rand_index(region, g) for g in gt_segs])),
                   VI=float(np.mean([variation_of_information(region, g) for g in gt_segs])),
                   _cntR=cntR, _sumR=sumR, _cntP=cntP, _sumP=sumP,
                   leakage=1 - float(np.mean([purity(region, g) for g in gt_segs])),
                   fragmentation=1 - float(np.mean([purity(g, region) for g in gt_segs])),
                   regions=int(len(np.unique(region))))
        P, R = _pr(cntR, sumR, cntP, sumP)
        row.update(F_b=float(_f(P, R)), P_b=float(P), R_b=float(R))
        self.rows.append(row)
        return {k: v for k, v in row.items() if not k.startswith('_')}

    def summary(self):
        r = self.rows
        s = lambda k: sum(x[k] for x in r)
        mean = lambda k: float(np.mean([x[k] for x in r]))
        P, R = _pr(s('_cntR'), s('_sumR'), s('_cntP'), s('_sumP'))
        return dict(SC=s('_cov') / s('_area'), PRI=mean('PRI'), VI=mean('VI'),
                    F_b=float(_f(P, R)), P_b=float(P), R_b=float(R),
                    leakage=mean('leakage'), fragmentation=mean('fragmentation'),
                    regions=mean('regions'), n=len(r))

    def per_image(self):
        return [{k: v for k, v in x.items() if not k.startswith('_')} for x in self.rows]
