"""
make_dev_set.py: 개선 실험용 소규모 개발 셋(dev set)을 BSDS500 val에서 골라 edge/dev_set.json 으로 고정

- test(200장)는 최종 평가에 한 번만 쓰고, 개선 과정은 이 dev set으로만 확인 (test에 맞춰 튜닝하는 것 방지)
- 복잡도(사람 GT의 평균 구역 수)로 val 100장을 정렬해 5구간으로 나누고, 구간마다 4장씩 무작위(seed 고정) → 20장
  → 단순한 사진부터 복잡한 사진까지 고르게 포함
- 한 번 만든 뒤에는 바꾸지 않음 (다시 만들려면 --force). 실험끼리 비교가 가능하려면 셋이 고정돼야 함

사용: python edge/make_dev_set.py
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbload import load_nb, HERE

parser = argparse.ArgumentParser()
parser.add_argument('--n-per-bin', type=int, default=4)
parser.add_argument('--bins', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--force', action='store_true', help='이미 있는 dev_set.json 덮어쓰기')
args = parser.parse_args()

DEV_JSON = os.path.join(HERE, 'dev_set.json')
if os.path.exists(DEV_JSON) and not args.force:
    raise SystemExit(f'{DEV_JSON} 이 이미 있음. 실험 비교를 위해 고정해 두는 파일이라 덮어쓰려면 --force')

ns = load_nb()
np, plt, cv2 = ns['np'], ns['plt'], ns['cv2']

ids = ns['val_ids']
complexity = {i: float(np.mean([s.max() for s in ns['load_gt'](i, 'val')[0]])) for i in ids}
ordered = sorted(ids, key=complexity.get)
rng = np.random.default_rng(args.seed)
dev = []
for b, chunk in enumerate(np.array_split(ordered, args.bins)):
    pick = sorted(rng.choice(chunk, args.n_per_bin, replace=False).tolist(), key=complexity.get)
    dev += pick
    print(f'구간 {b + 1}: GT 구역 {complexity[chunk[0]]:.1f}~{complexity[chunk[-1]]:.1f} → {pick}')

json.dump({
    'split': 'val',
    'ids': dev,
    'gt_segments_mean': {i: round(complexity[i], 1) for i in dev},
    'selection': f'val 100장을 GT 평균 구역 수로 정렬 → {args.bins}구간 × {args.n_per_bin}장 무작위 (seed={args.seed})',
}, open(DEV_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'\n{len(dev)}장 → {DEV_JSON}')

# 한눈에 보기 (단순 → 복잡 순)
out_dir = os.path.join(ns['OUT'], 'dev')
os.makedirs(out_dir, exist_ok=True)
cols = args.n_per_bin
fig, axes = plt.subplots(args.bins, cols, figsize=(3.2 * cols, 2.4 * args.bins))
for ax, i in zip(axes.ravel(), dev):
    ax.imshow(ns['bgr2rgb'](ns['load_image'](i, 'val')))
    ax.set_title(f'{i} (GT {complexity[i]:.0f} seg)', fontsize=9)
    ax.axis('off')
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'dev_set.png'), dpi=80)
print('미리보기:', os.path.join(out_dir, 'dev_set.png'))
