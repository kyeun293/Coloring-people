"""
unified_data.py: BSDS500과 PASCAL-Context를 같은 형식으로 맞춘 데이터셋 (담당: Taein)

만들기 (로컬, Colab 모두):
  python edge/data/unified_data.py              # 없는 원본은 자동 다운로드 후 변환 (PASCAL은 edge/data/splits 목록 그대로)
  python edge/data/unified_data.py --resample --pascal-frac 0.5 --seed 0   # 목록을 새로 뽑을 때만

결과 폴더 (기본 edge/edge_work/data/unified, 환경변수 EDGE_WORK로 바꿀 수 있음):
  <dataset>/images/<split>/<id>.jpg
  <dataset>/gt/<split>/<id>.npz
      seg  (주석자, H, W) int32   구역 번호 (1부터). B(segmentation) 평가용
      bnd  (주석자, H, W) bool    경계. A(edge) 학습과 평가용
      cls  (H, W) uint16          PASCAL만: 원래 클래스 번호 (1~459, labels.txt)
  splits/<dataset>_<split>.txt  이미지 id 목록
  meta.json                     만든 설정, 장 수, 출처

  dataset = bsds | pascal,  split = train | val | test

GT 규칙:
  BSDS500   seg = groundTruth[k]['Segmentation'], bnd = groundTruth[k]['Boundaries'] (주석자 4~9명, 원본 그대로)
  PASCAL    주석자 1명. LabelMap(클래스 지도)에서
            seg = 같은 클래스라도 떨어져 있으면 다른 구역 (8-연결 연결 요소). 'unknown'(431)도 일반 구역
            bnd = seg에서 BSDS와 똑같은 규칙(seg2bdry + thin)으로 만든 1픽셀 경계
  PASCAL split = COB 논문(ECCV 2016)의 VOC train / val / test에서 각각 frac 비율만큼 고정 seed로 뽑음

불러오기:
  import sys; sys.path.insert(0, 'edge/data')
  from unified_data import list_ids, load_sample
  img, seg, bnd = load_sample('pascal', 'train', '2008_000008')   # img: BGR uint8
"""
import argparse, json, os, random, shutil, subprocess, sys, tarfile, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                      # edge/data
FIXED_SPLITS = os.path.join(HERE, 'splits')                             # git에 올린 고정 split 목록 (있으면 그대로 사용)
WORK = os.environ.get('EDGE_WORK', os.path.join(os.path.dirname(HERE), 'edge_work'))
RAW = os.path.join(WORK, 'data')
UNI = os.path.join(RAW, 'unified')
SPLITS = ('train', 'val', 'test')

BSDS_URL = 'https://github.com/BIDS/BSDS500.git'
PC_ANN_URL = 'https://cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz'
PC_ANN_MD5 = 'df034edb2c12aa7d33b42b20bb1796e3'
PC_SPLIT_URL = 'https://data.vision.ee.ethz.ch/kmaninis/share/COB/Downloads/PASCALContext_splits.tgz'
VOC_URL = 'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar'


# ── 불러오기 (A, B가 쓰는 함수) ──────────────────────────
def list_ids(dataset, split, root=UNI):
    with open(os.path.join(root, 'splits', f'{dataset}_{split}.txt')) as f:
        return f.read().split()


def load_sample(dataset, split, img_id, root=UNI):
    """(img BGR uint8 (H,W,3), seg int32 (주석자,H,W), bnd bool (주석자,H,W))"""
    import cv2
    img = cv2.imread(os.path.join(root, dataset, 'images', split, f'{img_id}.jpg'))
    gt = np.load(os.path.join(root, dataset, 'gt', split, f'{img_id}.npz'))
    return img, gt['seg'], gt['bnd']


# ── GT 변환 규칙 ────────────────────────────────────────
def seg2bdry(seg):
    """구역 지도 → 1픽셀 경계. BSDS 공식 seg2bdry(seg, 'imageSize') 뒤에 thin을 한 것으로,
    BSDS500의 groundTruth 'Boundaries'를 'Segmentation'에서 정확히 재현함 (주석 2,696개 전부 픽셀 단위 일치 확인)"""
    from skimage.morphology import thin
    seg = np.asarray(seg)
    tx, ty = seg.shape
    b = np.zeros((2 * tx + 1, 2 * ty + 1), bool)                  # 2배 해상도 격자 (픽셀 사이에 경계 칸)
    v = np.zeros(seg.shape, bool); v[:-1] = seg[:-1] != seg[1:]
    h = np.zeros(seg.shape, bool); h[:, :-1] = seg[:, :-1] != seg[:, 1:]
    b[2::2, 1:-1:2] = v
    b[1:-1:2, 2::2] = h
    b[2:-1:2, 2:-1:2] = h[:-1, :-1] | h[1:, :-1] | v[:-1, :-1] | v[:-1, 1:]
    b[0], b[:, 0], b[-1], b[:, -1] = b[1], b[:, 1], b[-2], b[:, -2]
    return thin(b[2::2, 2::2])


def connected_regions(label_map):
    """클래스 지도 → 구역 지도: 같은 클래스라도 떨어져 있으면 다른 구역 (8-연결), 번호는 1부터"""
    from scipy import ndimage
    out = np.zeros(label_map.shape, np.int32)
    n = 0
    for c in np.unique(label_map):
        lab, k = ndimage.label(label_map == c, structure=np.ones((3, 3), int))
        out[lab > 0] = lab[lab > 0] + n
        n += k
    return out


# ── 원본 다운로드 ───────────────────────────────────────
def download(url, path):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print('다운로드:', url)
    tmp = path + '.part'
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


def md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _extract_all(t, dst):
    if hasattr(tarfile, 'data_filter'):          # Python 3.12+: 안전한 압축 풀기 (경로 탈출 차단)
        t.extractall(dst, filter='data')
    else:
        t.extractall(dst)


def _extract_one(t, m, dst):
    if hasattr(tarfile, 'data_filter'):
        t.extract(m, dst, filter='data')
    else:
        t.extract(m, dst)


def get_bsds():
    root = os.path.join(RAW, 'BSDS500', 'BSDS500', 'data')
    if not os.path.isdir(root):
        print('다운로드: BSDS500')
        subprocess.run(['git', 'clone', '-q', '--depth', '1', BSDS_URL, os.path.join(RAW, 'BSDS500')], check=True)
    return root


def get_pascal_ann():
    pc = os.path.join(RAW, 'pascal_context')
    ann = os.path.join(pc, 'ann', 'trainval')
    if not os.path.isdir(ann):
        tgz = download(PC_ANN_URL, os.path.join(pc, 'dl_ann', 'trainval.tar.gz'))
        if md5(tgz) != PC_ANN_MD5:
            raise SystemExit(f'MD5 불일치: {tgz} (파일을 지우고 다시 실행)')
        with tarfile.open(tgz) as t:
            _extract_all(t, os.path.join(pc, 'ann'))
    split_dir = os.path.join(pc, 'split', 'PASCALContext_splits')
    if not os.path.isdir(split_dir):
        tgz = download(PC_SPLIT_URL, os.path.join(pc, 'dl_split', 'PASCALContext_splits.tgz'))
        with tarfile.open(tgz) as t:
            _extract_all(t, os.path.join(pc, 'split'))
    return ann, split_dir, os.path.join(pc, 'ann', 'labels.txt')


def get_voc_images(need_ids):
    """VOC2010 JPEGImages 중 필요한 것만 확보 (없으면 1.3GB tar를 받아 필요한 파일만 풂)"""
    pc = os.path.join(RAW, 'pascal_context')
    jpg_dir = os.path.join(pc, 'voc', 'VOCdevkit', 'VOC2010', 'JPEGImages')
    missing = [i for i in need_ids if not os.path.exists(os.path.join(jpg_dir, f'{i}.jpg'))]
    if missing:
        tar_path = download(VOC_URL, os.path.join(pc, 'dl_voc', 'VOCtrainval_03-May-2010.tar'))
        want = {f'VOCdevkit/VOC2010/JPEGImages/{i}.jpg' for i in missing}
        print(f'압축 풀기: 이미지 {len(want)}장')
        with tarfile.open(tar_path) as t:
            for m in t:
                if m.name in want:
                    _extract_one(t, m, os.path.join(pc, 'voc'))
    return jpg_dir


# ── 변환 ────────────────────────────────────────────────
def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def build_bsds(root_out):
    import scipy.io as sio
    src = get_bsds()
    counts = {}
    for split in SPLITS:
        ids = sorted(f[:-4] for f in os.listdir(os.path.join(src, 'groundTruth', split)) if f.endswith('.mat'))
        img_dir = os.path.join(root_out, 'bsds', 'images', split)
        os.makedirs(img_dir, exist_ok=True)
        for i in ids:
            shutil.copyfile(os.path.join(src, 'images', split, f'{i}.jpg'), os.path.join(img_dir, f'{i}.jpg'))
            g = sio.loadmat(os.path.join(src, 'groundTruth', split, f'{i}.mat'))['groundTruth'][0]
            seg = np.stack([g[k]['Segmentation'][0, 0] for k in range(len(g))]).astype(np.int32)
            bnd = np.stack([g[k]['Boundaries'][0, 0] for k in range(len(g))]).astype(bool)
            save_npz(os.path.join(root_out, 'bsds', 'gt', split, f'{i}.npz'), seg=seg, bnd=bnd)
        write_split(root_out, 'bsds', split, ids)
        counts[split] = len(ids)
        print(f'bsds {split}: {len(ids)}장')
    return counts


def build_pascal(root_out, frac, seed, use_fixed=True):
    import cv2
    import scipy.io as sio
    ann, split_dir, labels_txt = get_pascal_ann()
    cob = {'train': 'voc_train', 'val': 'voc_val', 'test': 'voc_test'}
    chosen = {}
    for split in SPLITS:
        fixed = os.path.join(FIXED_SPLITS, f'pascal_{split}.txt')
        if use_fixed and os.path.exists(fixed):      # 팀 공용 목록 (모두 같은 이미지를 쓰도록)
            chosen[split] = open(fixed).read().split()
            continue
        ids = sorted(open(os.path.join(split_dir, f'{cob[split]}.txt')).read().split())
        k = int(round(len(ids) * frac))
        chosen[split] = sorted(random.Random(f'{seed}-{split}').sample(ids, k))
    jpg_dir = get_voc_images([i for v in chosen.values() for i in v])
    counts = {}
    for split, ids in chosen.items():
        img_dir = os.path.join(root_out, 'pascal', 'images', split)
        os.makedirs(img_dir, exist_ok=True)
        for n, i in enumerate(ids):
            src_jpg = os.path.join(jpg_dir, f'{i}.jpg')
            cls = sio.loadmat(os.path.join(ann, f'{i}.mat'))['LabelMap'].astype(np.uint16)
            if cv2.imread(src_jpg).shape[:2] != cls.shape:
                raise SystemExit(f'이미지와 주석 크기 다름: {i}')
            shutil.copyfile(src_jpg, os.path.join(img_dir, f'{i}.jpg'))
            seg = connected_regions(cls)
            save_npz(os.path.join(root_out, 'pascal', 'gt', split, f'{i}.npz'),
                     seg=seg[None], bnd=seg2bdry(seg)[None], cls=cls)
            if n % 500 == 0:
                print(f'  pascal {split} {n}/{len(ids)}', flush=True)
        write_split(root_out, 'pascal', split, ids)
        counts[split] = len(ids)
        print(f'pascal {split}: {len(ids)}장')
    shutil.copyfile(labels_txt, os.path.join(root_out, 'pascal', 'labels.txt'))
    return counts


def write_split(root_out, dataset, split, ids):
    os.makedirs(os.path.join(root_out, 'splits'), exist_ok=True)
    with open(os.path.join(root_out, 'splits', f'{dataset}_{split}.txt'), 'w') as f:
        f.write('\n'.join(ids) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pascal-frac', type=float, default=0.5, help='COB split마다 뽑을 비율')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--only', choices=['bsds', 'pascal'], default=None)
    ap.add_argument('--out', default=UNI)
    ap.add_argument('--resample', action='store_true',
                    help='edge/data/splits 고정 목록 대신 --pascal-frac, --seed로 새로 뽑기')
    a = ap.parse_args()

    meta_path = os.path.join(a.out, 'meta.json')
    meta = json.load(open(meta_path, encoding='utf-8')) if os.path.exists(meta_path) else {}
    if a.only != 'pascal':
        meta['bsds'] = dict(counts=build_bsds(a.out), source=BSDS_URL, annotators='4~9 (원본 그대로)')
    if a.only != 'bsds':
        meta['pascal'] = dict(counts=build_pascal(a.out, a.pascal_frac, a.seed, not a.resample),
                              frac=a.pascal_frac, seed=a.seed, fixed_split_list=not a.resample,
                              split='COB (ECCV 2016) voc_train/voc_val/voc_test 각각에서 frac 비율 무작위 추출',
                              seg='LabelMap의 8-연결 연결 요소 (unknown 포함 모든 클래스)', bnd='seg2bdry(seg)',
                              source=[PC_ANN_URL, PC_SPLIT_URL, VOC_URL])
    json.dump(meta, open(meta_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('완료:', a.out)
