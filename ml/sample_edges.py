"""
sample_edges.py — 엣지 검출 비교 샘플 이미지 만들기
====================================================
BSDS500 테스트 이미지마다 한 줄씩: 원본 | 정답(사람 주석) | TEED 그대로 | 후처리 | 후처리 + 선 잇기 | Sobel

실행:
  python3 ml/sample_edges.py                       # 기본 샘플 6장
  python3 ml/sample_edges.py 189006 246009 ...     # 원하는 테스트 이미지 번호
  python3 ml/sample_edges.py --model edge_v2       # 다른 모델 (ml/models/<model>.pt)
출력: ml/samples/edge_samples.png (기본 모델) / ml/samples/edge_samples_<model>.png
"""

import os, sys
import numpy as np
import cv2
import scipy.io
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_model import TED
from train_edge import BSDS_DIR, MODEL_DIR, DEVICE, predict, sobel_baseline

DEFAULT_IDS = ['189006', '246009', '230098', '100039', '108004', '226033']
OUT_PATH = 'ml/samples/edge_samples.png'
FONT_PATHS = ['/System/Library/Fonts/AppleSDGothicNeo.ttc', '/System/Library/Fonts/Supplemental/AppleGothic.ttf']


def skeletonize(mask):
    """Zhang-Suen 세선화: 굵은 선을 가운데 1px 뼈대로 깎음"""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            P = [img[1:-1, 1:-1], img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:],
                 img[2:, 1:-1], img[2:, :-2], img[1:-1, :-2], img[:-2, :-2]]   # P1, P2..P9 (시계방향)
            nb = P[1:] + [P[1]]
            B = sum(P[1:9])
            A = sum(((nb[i] == 0) & (nb[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = (P[1] * P[3] * P[5] == 0) & (P[3] * P[5] * P[7] == 0)
            else:
                c = (P[1] * P[3] * P[7] == 0) & (P[1] * P[5] * P[7] == 0)
            rm = (P[0] == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any():
                img[1:-1, 1:-1][rm] = 0; changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


LINK_DIST  = 20   # 이을 최대 거리(px)
LINK_ANGLE = 70   # 끝점↔끝점: 두 선의 진행 방향 허용 각도(°)
RAY_ANGLE  = 20   # 끝점→선: 진행 방향 기준 부채꼴 반각(°)
LINK_BACK  = 6    # 끝점 방향을 잴 때 선을 따라 거슬러 갈 거리(px)


def link_endpoints(skel, max_dist=LINK_DIST, max_angle=LINK_ANGLE, ray_angle=RAY_ANGLE, back=LINK_BACK):
    """
    끊긴 선 잇기 (가까운 것부터, 끝점당 한 번)
      1) 끝점↔끝점: max_dist 안이고 두 선이 서로를 향해 뻗어 있으면(max_angle 안) 직선 연결
      2) 끝점→선: 남은 끝점에서 진행 방향 ±ray_angle 부채꼴로 max_dist 안에 다른 선이 있으면 거기까지 연결 (T자 이음)
    반환: (이은 뒤 뼈대, 새로 그은 픽셀)
    """
    sk = skel.astype(np.uint8)
    h, w = sk.shape
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    ends = [tuple(p) for p in np.argwhere((sk == 1) & (nb == 1))]
    _, comp, stats, _ = cv2.connectedComponentsWithStats(sk, connectivity=8)

    def trace(y, x, steps):
        """끝점에서 선을 따라 steps 픽셀 거슬러 간 경로"""
        path = [(y, x)]; seen = {(y, x)}
        for _ in range(steps):
            cy, cx = path[-1]
            nxt = next(((cy + dy, cx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                        if (dy or dx) and 0 <= cy + dy < h and 0 <= cx + dx < w
                        and sk[cy + dy, cx + dx] and (cy + dy, cx + dx) not in seen), None)
            if nxt is None: break
            seen.add(nxt); path.append(nxt)
        return path

    info = []
    for (y, x) in ends:
        path = trace(y, x, max(back, 2 * max_dist))
        cy, cx = path[min(back, len(path) - 1)]
        v = np.array([y - cy, x - cx], np.float32); n = np.linalg.norm(v)
        info.append((v / n if n else None, set(path)))   # (진행 방향, 자기 선의 가까운 부분 — 자기 자신에 잇지 않게)

    out = sk.copy(); used = set()
    def draw(p, q): cv2.line(out, p[::-1], q[::-1], 1, 1)

    # 1) 끝점 ↔ 끝점
    cos_lim = np.cos(np.deg2rad(max_angle)); pairs = []
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            di, dj = info[i][0], info[j][0]
            if di is None or dj is None: continue
            v = np.array(ends[j], np.float32) - np.array(ends[i], np.float32); d = np.linalg.norm(v)
            if d == 0 or d > max_dist or ends[j] in info[i][1]: continue
            if di @ (v / d) >= cos_lim and dj @ (-v / d) >= cos_lim:
                pairs.append((d, i, j))
    for d, i, j in sorted(pairs):
        if i in used or j in used: continue
        used.update((i, j)); draw(ends[i], ends[j])

    # 2) 끝점 → 선 (부채꼴 안에서 가장 가까운 선 픽셀)
    angles = np.deg2rad(np.linspace(-ray_angle, ray_angle, 9))
    for i, (y, x) in enumerate(ends):
        d0, own = info[i]
        if i in used or d0 is None: continue
        best = None
        for t in range(2, max_dist + 1):
            for a in angles:
                dy = d0[0] * np.cos(a) - d0[1] * np.sin(a); dx = d0[0] * np.sin(a) + d0[1] * np.cos(a)
                py, px = int(round(y + dy * t)), int(round(x + dx * t))
                if 0 <= py < h and 0 <= px < w and sk[py, px] and (py, px) not in own:
                    best = (py, px); break
            if best: break
        if best:
            used.add(i); draw((y, x), best)
    return out.astype(bool), (out > 0) & (sk == 0)


def clean_edges(prob, thresh=0.6, min_len=30, link=True):
    """
    사람이 그린 것처럼: 확실한 선만 → 작은 틈 메우기 → 1px 뼈대 → (선 끝 잇기) → 짧은 조각 제거
    반환: (최종 선, 선 잇기로 새로 그은 픽셀)
    """
    mask = (prob > thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    skel = skeletonize(mask)
    added = np.zeros_like(skel)
    if link:
        skel, added = link_endpoints(skel)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(skel.astype(np.uint8), connectivity=8)
    keep = (stats[lab, cv2.CC_STAT_AREA] >= min_len) & (lab > 0)
    return keep, added & keep


def overlay(lines, added):
    """흰 바탕 검은 선 + 새로 이은 부분은 빨강"""
    k = np.ones((2, 2), np.uint8)
    img = gray_to_rgb(cv2.dilate(lines.astype(np.float32), k))
    img[cv2.dilate(added.astype(np.uint8), k) > 0] = (230, 30, 30)
    return img


def gray_to_rgb(x):
    """0~1 경계 강도 → 흰 바탕에 검은 선"""
    g = (255 * (1 - np.clip(x, 0, 1))).astype(np.uint8)
    return np.dstack([g, g, g])


def main():
    args = sys.argv[1:]
    name = 'edge'
    if '--model' in args:
        k = args.index('--model'); name = args[k + 1]; del args[k:k + 2]
    ids = args or DEFAULT_IDS
    out_path = OUT_PATH if name == 'edge' else OUT_PATH.replace('.png', f'_{name}.png')
    model = TED().to(DEVICE)
    model.load_state_dict(torch.load(f'{MODEL_DIR}/{name}.pt', map_location=DEVICE))
    model.eval()

    rows = []
    for name in ids:
        rgb = cv2.cvtColor(cv2.imread(f'{BSDS_DIR}/images/test/{name}.jpg'), cv2.COLOR_BGR2RGB)
        gts = scipy.io.loadmat(f'{BSDS_DIR}/groundTruth/test/{name}.mat')['groundTruth'][0]
        agree = np.mean([g['Boundaries'][0, 0].astype(np.float32) for g in gts], axis=0)
        gt = cv2.dilate(agree, np.ones((2, 2), np.uint8))       # 1px 선이 축소 후에도 보이게
        prob = predict(model, rgb)
        no_link, _ = clean_edges(prob, link=False)
        linked, added = clean_edges(prob, link=True)
        cells = [rgb, gray_to_rgb(gt), gray_to_rgb(prob), overlay(no_link, np.zeros_like(no_link)),
                 overlay(linked, added), gray_to_rgb(sobel_baseline(rgb))]
        if rgb.shape[0] > rgb.shape[1]:                          # 세로 사진은 가로 크기에 맞춤
            cells = [cv2.resize(c, (481, 321)) for c in cells]
        rows.append(np.hstack([cv2.copyMakeBorder(c, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(230, 230, 230))
                               for c in cells]))

    body = np.vstack(rows)
    header_h = 48
    canvas = Image.new('RGB', (body.shape[1], body.shape[0] + header_h), (255, 255, 255))
    canvas.paste(Image.fromarray(body), (0, header_h))
    draw = ImageDraw.Draw(canvas)
    font = next((ImageFont.truetype(p, 24) for p in FONT_PATHS if os.path.exists(p)), ImageFont.load_default())
    col_w = body.shape[1] // 6
    for i, t in enumerate(['원본', '정답 (사람)', 'TEED 그대로', '후처리', '후처리 + 선 잇기 (빨강)', 'Sobel']):
        tw = draw.textlength(t, font=font)
        draw.text((i * col_w + (col_w - tw) / 2, 10), t, fill=(20, 20, 20), font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    print(f'저장: {out_path}  ({canvas.size[0]}x{canvas.size[1]}, {len(ids)}장)')


if __name__ == '__main__':
    main()
