"""
dev_eval.py: 개발용 셋(edge/dev_set.json, BSDS500 val 20장)으로 방법/후처리 버전을 평가하고 기록

사용:
  python edge/dev_eval.py --tag v1                       # 기본: kmeans, sobel, canny, hed × 후처리 v1
  python edge/dev_eval.py --tag v2 --method hed --pp v1 v2 --note "hysteresis 추가"

결과 (edge/results/dev/):
  experiments.csv          실험 기록 누적 (실험마다 한 줄씩 추가 → 개선 과정 표)
  <tag>_per_image.csv      이미지별 점수
  <tag>_<method>_<pp>.png  dev set 전체의 칠한 예시 (눈으로 확인)

후처리 버전은 edge_coloring.py 의 POSTPROCESS 등록부에서 이름으로 고름.
딥러닝 모델(HED, PiDiNet, TEED, DexiNed) 확률 맵은 가중치가 고정이라 edge_work/cache/ 에 저장해 두고 재사용.
(Sobel/Canny는 코드를 고칠 수 있어서 매번 새로 계산)
"""
import argparse, json, os, sys, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbload import load_nb, HERE

parser = argparse.ArgumentParser()
parser.add_argument('--tag', required=True, help='실험 이름 (예: v1, v2-hysteresis)')
ALL_METHODS = ['kmeans', 'sobel', 'canny', 'hed', 'pidinet', 'teed', 'dexined']
parser.add_argument('--method', nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
parser.add_argument('--pp', nargs='+', default=['v1'], help='후처리 버전 (POSTPROCESS 등록부 이름)')
parser.add_argument('--note', default='', help='무엇을 바꿨는지 한 줄 메모')
parser.add_argument('--baseline', default='v1', help='채택 판정의 기준 후처리 버전 (같은 실행 안에서 비교)')
parser.add_argument('--no-images', action='store_true', help='결과 그림 생략 (빠르게 숫자만)')
args = parser.parse_args()

ns = load_nb()
np, pd, plt = ns['np'], ns['pd'], ns['plt']
dev = json.load(open(os.path.join(HERE, 'dev_set.json'), encoding='utf-8'))
split, ids = dev['split'], dev['ids']
OUT = os.path.join(ns['OUT'], 'dev')
CACHE = os.path.join(ns['WORK'], 'cache')
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
for p in args.pp:
    if p not in ns['POSTPROCESS']:
        raise SystemExit(f'후처리 버전 {p!r} 없음. 등록된 버전: {list(ns["POSTPROCESS"])}')


CACHED = ('hed', 'pidinet', 'teed', 'dexined')   # 가중치가 고정된 딥러닝 모델만 캐시 (Sobel/Canny는 코드를 고칠 수 있어서 매번 계산)


def edge_prob(det, img_id, img):
    if det not in CACHED:
        return ns['DETECTORS'][det](img)
    path = os.path.join(CACHE, f'{det}_{split}_{img_id}.npy')
    if os.path.exists(path):
        return np.load(path)
    prob = ns['DETECTORS'][det](img)
    np.save(path, prob)
    return prob


# (method, pp) 조합: kmeans는 자체 구역 분할이라 후처리 버전 없음
combos = [('kmeans', '-')] if 'kmeans' in args.method else []
combos += [(m, p) for m in args.method if m != 'kmeans' for p in args.pp]

rows, colored = [], {c: [] for c in combos}
for img_id in ids:
    img = ns['load_image'](img_id, split)
    segs, bnds = ns['load_gt'](img_id, split)
    probs = {}
    for method, pp in combos:
        t0 = time.perf_counter()
        if method == 'kmeans':
            region = ns['regions_kmeans_baseline'](img)
        else:
            if method not in probs:
                probs[method] = edge_prob(method, img_id, img)
                t0 = time.perf_counter()       # 검출 시간 제외, 후처리 시간만 (HED는 캐시라 공정 비교 불가)
            region = ns['POSTPROCESS'][pp](probs[method], **ns['PP'][method])
        ms = (time.perf_counter() - t0) * 1000
        rows.append(dict(id=img_id, method=method, pp=pp, ms=ms, **ns['evaluate_region'](region, segs, bnds)))
        if not args.no_images:
            colored[(method, pp)].append(ns['build_coloring_book'](img, region)['colored'])
    print('.', end='', flush=True)
print()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, f'{args.tag}_per_image.csv'), index=False)
cols = ['region_F', 'F', 'P', 'R', 'leakage', 'fragmentation', 'regions', 'ms']
summary = df.groupby(['method', 'pp'], sort=False)[cols].mean().round(3).reset_index()

# 실험 기록 누적
log_path = os.path.join(OUT, 'experiments.csv')
log = summary.assign(time=datetime.now().strftime('%Y-%m-%d %H:%M'), tag=args.tag, n=len(ids), note=args.note)
log = log[['time', 'tag', 'method', 'pp', 'n'] + cols + ['note']]
log.to_csv(log_path, mode='a', index=False, header=not os.path.exists(log_path), encoding='utf-8-sig')

print(f'\n[{args.tag}] dev set {len(ids)}장 ({split})  ※ ms = 후처리 시간')
print(summary.to_string(index=False))

# 채택 판정: 같은 검출기에서 baseline 후처리 대비 (규칙: Region F ↑ 이고 Boundary F 하락 ≤ 0.01)
MAX_F_DROP = 0.01
base = summary[summary.pp == args.baseline].set_index('method')
cand = summary[(summary.pp != args.baseline) & (summary.method.isin(base.index))]
if len(cand):
    print(f'\n채택 판정 (기준: {args.baseline}, 규칙: Region F ↑ 그리고 Boundary F 하락 ≤ {MAX_F_DROP})')
    for _, r in cand.iterrows():
        b = base.loc[r['method']]
        d_rf, d_f = r['region_F'] - b['region_F'], r['F'] - b['F']
        ok = d_rf > 0 and d_f >= -MAX_F_DROP
        print(f"  {r['method']:6s} {r['pp']:8s} Region F {d_rf:+.3f}  Boundary F {d_f:+.3f}  "
              f"leakage {r['leakage'] - b['leakage']:+.3f}  frag {r['fragmentation'] - b['fragmentation']:+.3f}"
              f"  → {'채택' if ok else '기각'}")

# 이전 실험과 비교 (같은 method+pp의 직전 기록 대비, 재현 확인용)
hist = pd.read_csv(log_path, encoding='utf-8-sig')
prev = hist[hist['tag'] != args.tag].groupby(['method', 'pp']).tail(1)
if len(prev) and 'region_F' in prev:
    m = summary.merge(prev, on=['method', 'pp'], suffixes=('', '_prev'))
    if len(m):
        print('\n직전 기록 대비 변화')
        for _, r in m.iterrows():
            print(f"  {r['method']:6s} {r['pp']:8s} vs {r['tag']}: Region F {r['region_F'] - r['region_F_prev']:+.3f}  "
                  f"Boundary F {r['F'] - r['F_prev']:+.3f}")

# dev set 전체 칠한 예시 그림
if not args.no_images:
    for (method, pp), ims in colored.items():
        n_cols = 5
        n_rows = (len(ims) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.4 * n_rows))
        for ax in axes.ravel():
            ax.axis('off')
        sub = df[(df.method == method) & (df.pp == pp)].reset_index(drop=True)
        for k, (ax, im) in enumerate(zip(axes.ravel(), ims)):
            ax.imshow(im)
            ax.set_title(f"{sub.id[k]}  F {sub.F[k]:.2f} / leak {sub.leakage[k]:.2f}", fontsize=8)
        fig.suptitle(f'{args.tag}: {method} + {pp}', fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, f'{args.tag}_{method}_{pp}.png'.replace('_-.png', '.png')), dpi=70)
        plt.close(fig)

print('\n저장 위치:', OUT)
