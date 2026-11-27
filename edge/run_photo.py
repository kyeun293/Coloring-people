"""
run_photo.py: 내 사진 한 장에 Sobel / Canny / HED + 후처리를 적용해서 결과를 edge/results/ 에 저장

사용:
  python edge/run_photo.py 사진.jpg                      # 세 방법 모두
  python edge/run_photo.py 사진.jpg --method hed         # 하나만 (sobel | canny | hed)
  python edge/run_photo.py a.jpg b.jpg c.jpg            # 여러 장

저장 파일 (사진 이름 끝 9자리 기준):
  <이름>_compare.png    원본 / edge 맵 / 색칠판 / 칠한 예시
  <이름>_hed_steps.png  HED 후처리 단계별 (HED를 돌린 경우)

알고리즘은 edge_coloring.py(노트북 소스)의 0~3장을 그대로 불러와서 씀 → 노트북을 고치면 여기도 반영됨
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbload import load_nb

parser = argparse.ArgumentParser()
parser.add_argument('photos', nargs='+')
parser.add_argument('--method', nargs='+', default=['sobel', 'canny', 'hed'], choices=['sobel', 'canny', 'hed'])
parser.add_argument('--width', type=int, default=480, help='처리 크기 (웹 앱과 같은 480)')
args = parser.parse_args()

ns = load_nb()
cv2, np, plt, OUT = ns['cv2'], ns['np'], ns['plt'], ns['OUT']

for photo in args.photos:
    img = cv2.imdecode(np.fromfile(photo, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print('읽을 수 없음:', photo)
        continue
    stem = os.path.splitext(os.path.basename(photo))[0][-9:]
    h, w = img.shape[:2]
    img = cv2.resize(img, (args.width, round(h * args.width / w)), interpolation=cv2.INTER_AREA)

    cols = {}
    for det in args.method:
        prob = ns['DETECTORS'][det](img)
        region, steps = ns['edges_to_regions'](prob, **ns['PP'][det], return_steps=True)
        cols[det] = dict(prob=prob, book=ns['build_coloring_book'](img, region), steps=steps)

    # 비교 그림 (행: edge 맵 / 색칠판 / 칠한 예시)
    n = len(cols) + 1
    fig, ax = plt.subplots(3, n, figsize=(4 * n, 3 * 4 * img.shape[0] / img.shape[1] + 1), squeeze=False)
    for a in ax.ravel():
        a.axis('off')
    ax[0, 0].imshow(ns['bgr2rgb'](img)); ax[0, 0].set_title('Original photo')
    ax[1, 0].imshow(ns['palette_strip'](next(iter(cols.values()))['book']['palette'])); ax[1, 0].set_title('Palette (k-means, 7)')
    for j, (det, c) in enumerate(cols.items(), 1):
        name = det.upper() if det == 'hed' else det.capitalize()
        ax[0, j].imshow(c['prob'], cmap='gray'); ax[0, j].set_title(f'{name}: edge map')
        ax[1, j].imshow(c['book']['board']); ax[1, j].set_title(f'{name} + postprocess ({c["book"]["n_regions"]} regions)')
        ax[2, j].imshow(c['book']['colored']); ax[2, j].set_title(f'{name}: colored example')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f'{stem}_compare.png'), dpi=80)
    plt.close(fig)

    if 'hed' in cols:   # HED 후처리 단계별
        c = cols['hed']
        vis = [c['prob']] + [ns['random_colors'](v) if 'regions' in k else v for k, v in c['steps'].items()]
        vis.append(c['book']['line'])
        ns['show'](vis, ['HED prob'] + list(c['steps'].keys()) + ['6 line'], cols=4, save=f'{stem}_hed_steps.png')

    print(f'{os.path.basename(photo)} → ' + ', '.join(f'{d}: {c["book"]["n_regions"]} regions' for d, c in cols.items()))

print('저장 위치:', OUT)
