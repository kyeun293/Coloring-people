"""
model_table.py: 모델별 "기본 모델 → +후처리 v1 → +후처리 v2 → Ground Truth" 결과 사진 표 (HTML)

사용:
  python edge/model_table.py                       # dev set에서 복잡도 구간별 1장씩 5장
  python edge/model_table.py --ids 304034 101087   # 원하는 val 이미지
  python edge/model_table.py --dev-tag models      # 위쪽 요약표에 쓸 dev_eval 결과 (results/dev/<tag>_per_image.csv)

출력: edge/results/model_table.html (이미지 포함 단일 파일, 브라우저로 열기)

- 기본 모델 칸: edge 확률 맵 (검은 선 = 강한 edge). 점수는 "임계값 + 얇게"만 한 선의 Boundary F
- 후처리 칸: 팔레트 색으로 칠한 색칠판 (색이 섞여 칠해진 곳 = 새는 곳). 점수는 Region F / Boundary F / Leakage / Fragmentation / 구역 수
- Ground Truth 칸: 사람 주석 1명분(구역 수가 중간인 주석자)을 같은 방식으로 칠한 것
"""
import argparse, base64, html, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbload import load_nb, HERE

parser = argparse.ArgumentParser()
parser.add_argument('--ids', nargs='+', default=None)
parser.add_argument('--models', nargs='+', default=['canny', 'sobel', 'hed', 'pidinet', 'teed', 'dexined'])
parser.add_argument('--pp', nargs=2, default=['v1', 'v2'], help='비교할 후처리 두 개')
parser.add_argument('--dev-tag', default='models')
args = parser.parse_args()

ns = load_nb()
np, cv2, pd = ns['np'], ns['cv2'], ns['pd']
dev = json.load(open(os.path.join(HERE, 'dev_set.json'), encoding='utf-8'))
split = dev['split']
ids = args.ids or [dev['ids'][k] for k in (1, 5, 9, 12, 18)]   # 복잡도 5구간에서 1장씩
CACHE = os.path.join(ns['WORK'], 'cache')

INFO = {
    'canny':   ('Canny', '규칙 기반 (강의)'),
    'sobel':   ('Sobel', '규칙 기반 (강의)'),
    'hed':     ('HED', 'BSDS ⚠️'),
    'pidinet': ('PiDiNet', 'BSDS ⚠️'),
    'teed':    ('TEED', 'BIPED'),
    'dexined': ('DexiNed', 'BIPED'),
    'kmeans':  ('팀 원본 k-means', '규칙 기반'),
}


def edge_prob(m, img_id, img):
    path = os.path.join(CACHE, f'{m}_{split}_{img_id}.npy')
    if os.path.exists(path):
        return np.load(path)
    p = ns['DETECTORS'][m](img)
    if m in ('hed', 'pidinet', 'teed', 'dexined'):
        os.makedirs(CACHE, exist_ok=True)
        np.save(path, p)
    return p


def base_f(prob, m, bnds):
    """기본 모델 성능: 임계값 + 얇게(skeletonize)만 한 선의 Boundary F"""
    line = ns['skeletonize'](prob >= ns['PP'][m]['thresh'])
    return ns['boundary_f'](line, bnds)[2]


def img_tag(arr, width=230):
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(buf).decode()}" style="width:{width}px">'


def fmt_metrics(r):
    return (f'<b>Region F {r["region_F"]:.3f}</b><br>Boundary F {r["F"]:.3f}<br>'
            f'Leak {r["leakage"]:.3f} · Frag {r["fragmentation"]:.3f}<br>구역 {r["regions"]}')


# ── 위쪽 요약표: dev set 20장 평균 ──────────────────────
summary_html = ''
per_image = os.path.join(ns['OUT'], 'dev', f'{args.dev_tag}_per_image.csv')
if os.path.exists(per_image):
    df = pd.read_csv(per_image, dtype={'id': str})
    base_rows = []
    for img_id in dev['ids']:
        img = ns['load_image'](img_id, split)
        _, bnds = ns['load_gt'](img_id, split)
        for m in args.models:
            base_rows.append(dict(method=m, base_F=base_f(edge_prob(m, img_id, img), m, bnds)))
    base = pd.DataFrame(base_rows).groupby('method')['base_F'].mean()
    rows = []
    best = {pp: df[df.pp == pp]['region_F'].groupby(df.method).mean().max() for pp in args.pp}
    for m in ['kmeans'] + args.models:
        name, data = INFO[m]
        if m == 'kmeans':
            k = df[df.method == 'kmeans'].mean(numeric_only=True)
            rows.append(f'<tr><td>{name}</td><td>{data}</td><td class=c>–</td>'
                        f'<td class=c colspan=2>{k.region_F:.3f} / {k.F:.3f} / {k.leakage:.3f} / {k.fragmentation:.3f} / {k.regions:.1f}'
                        f' <span class=muted>(구역 방식이라 후처리 없음)</span></td></tr>')
            continue
        tds = [f'<td>{name}</td><td>{data}</td><td class=c>{base[m]:.3f}</td>']
        for pp in args.pp:
            g = df[(df.method == m) & (df.pp == pp)].mean(numeric_only=True)
            mark = ' best' if abs(g.region_F - best[pp]) < 1e-9 else ''
            tds.append(f'<td class="c{mark}"><b>{g.region_F:.3f}</b> / {g.F:.3f} / {g.leakage:.3f} / '
                       f'{g.fragmentation:.3f} / {g.regions:.1f}</td>')
        rows.append('<tr>' + ''.join(tds) + '</tr>')
    summary_html = f'''
<h2>dev set {len(dev["ids"])}장 평균 (BSDS500 {split})</h2>
<table class=sum>
<tr><th>모델</th><th>학습 데이터</th><th>기본 모델<br>Boundary F</th>
<th>+ 후처리 {args.pp[0]}<br><span class=muted>Region F / Boundary F / Leak / Frag / 구역</span></th>
<th>+ 후처리 {args.pp[1]}<br><span class=muted>Region F / Boundary F / Leak / Frag / 구역</span></th></tr>
{''.join(rows)}
</table>
<p class=muted>Region F(주 지표)가 각 열에서 가장 높은 칸은 초록색. ⚠️ HED·PiDiNet은 BSDS500 train/val로 학습된 모델이라 val 점수가 실제보다 좋게 나올 수 있음.
후처리 파라미터는 튜닝 없이 모두 HED 기본값(thresh=0.3, closing 3, 최소 구역 0.2%).</p>'''

# ── 이미지별 결과 사진 표 ─────────────────────────────
sections = []
for img_id in ids:
    img = ns['load_image'](img_id, split)
    segs, bnds = ns['load_gt'](img_id, split)
    k = int(np.argsort([s.max() for s in segs])[len(segs) // 2])          # 구역 수가 중간인 주석자
    gt_book = ns['build_coloring_book'](img, segs[k] - 1)
    gt_cell = (f'<td rowspan={len(args.models)} class=gt>{img_tag(gt_book["colored"])}'
               f'<div class=m>주석자 {k + 1}/{len(segs)}<br>구역 {gt_book["n_regions"]}</div>'
               f'<div class=orig>원본 사진<br>{img_tag(ns["bgr2rgb"](img), 160)}</div></td>')
    trs = []
    for i, m in enumerate(args.models):
        prob = edge_prob(m, img_id, img)
        name, data = INFO[m]
        tds = [f'<th class=rowh>{name}<br><span class=muted>{data}</span></th>',
               f'<td>{img_tag(1 - prob)}<div class=m>Boundary F {base_f(prob, m, bnds):.3f}<br>'
               f'<span class=muted>(임계값+얇게)</span></div></td>']
        for pp in args.pp:
            region = ns['POSTPROCESS'][pp](prob, **ns['PP'][m])
            r = ns['evaluate_region'](region, segs, bnds)
            tds.append(f'<td>{img_tag(ns["build_coloring_book"](img, region)["colored"])}'
                       f'<div class=m>{fmt_metrics(r)}</div></td>')
        if i == 0:
            tds.append(gt_cell)
        trs.append('<tr>' + ''.join(tds) + '</tr>')
    gt_n = np.mean([s.max() for s in segs])
    sections.append(f'''
<h2>{img_id} <span class=muted>(GT 평균 구역 {gt_n:.0f}개)</span></h2>
<table class=grid>
<tr><th></th><th>① 기본 모델 (edge 맵)</th><th>② + 후처리 {args.pp[0]}</th><th>③ + 후처리 {args.pp[1]}</th><th>④ Ground Truth</th></tr>
{''.join(trs)}
</table>''')
    print('.', end='', flush=True)
print()

page = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>Edge 모델 비교표</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", "Malgun Gothic", sans-serif; margin: 24px; color: #222; background: #fafafa; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin-top: 36px; }}
table {{ border-collapse: collapse; background: #fff; }}
th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; text-align: center; font-size: 12px; }}
th {{ background: #f0f0f0; }}
.sum td, .sum th {{ font-size: 13px; padding: 6px 10px; }}
.sum td.c {{ font-variant-numeric: tabular-nums; }}
.best {{ background: #e6f4ea; }}
.rowh {{ width: 90px; }}
.m {{ margin-top: 4px; line-height: 1.45; font-variant-numeric: tabular-nums; }}
.muted {{ color: #888; font-weight: normal; font-size: 11px; }}
.gt {{ background: #fffbea; }}
.orig {{ margin-top: 14px; color: #666; }}
</style></head><body>
<h1>Edge 모델별 색칠공부판 비교: 기본 모델 → 후처리 {args.pp[0]} → 후처리 {args.pp[1]} → Ground Truth</h1>
<p class=muted>후처리 칸은 구역마다 팔레트 색으로 칠한 결과예요. 한 구역에 여러 물체가 같은 색으로 칠해져 있으면 색이 새는 것(Leakage),
한 물체가 여러 조각으로 나뉘어 있으면 쪼개진 것(Fragmentation)이에요.</p>
{summary_html}
{''.join(sections)}
</body></html>'''

out = os.path.join(ns['OUT'], 'model_table.html')
open(out, 'w', encoding='utf-8').write(page)
print('저장:', out, f'({os.path.getsize(out) / 1e6:.1f} MB)')
