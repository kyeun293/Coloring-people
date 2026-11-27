# 통일 데이터셋: BSDS500 + PASCAL-Context (담당: Taein)

edge detection(A)과 segmentation(B)이 **같은 데이터, 같은 형식, 같은 split**을 쓰도록 정리한 폴더예요.
데이터 파일 자체는 용량(약 700MB)과 라이선스 때문에 git에 넣지 않았어요. 스크립트를 실행하면 원래 배포처에서 받아서 **모두에게 똑같은 데이터**가 만들어져요.

| 파일 | 내용 |
|---|---|
| `unified_data.py` | 다운로드 → 변환 스크립트 + 불러오기 함수 (`list_ids`, `load_sample`) |
| `edge_datasets.py` | **바로 학습·평가에 쓰는 PyTorch Dataset** (학습 라벨, augmentation, 모델별 정규화, RCF 손실) |
| `evaluation.py` | **표준 평가 코드**: A용 ODS/OIS/AP, B용 SC/PRI/VI/F_b (+ 색칠공부 보조 지표) |
| `splits/*.txt` | **팀 공용 고정 split 목록** (어떤 이미지를 train/val/test로 쓰는지). 이 목록을 바꾸지 마세요 |

**한눈에 보기**
- 데이터: BSDS500 500장 (200/100/200) + PASCAL-Context 5,052장 (2,499/1,304/1,249, COB split의 50%)
- GT: `seg` 구역 지도 (B), `bnd` 1픽셀 경계 (A). PASCAL은 클래스 지도를 연결 요소로 나눠 만듦
- 학습 라벨: 주석자 비율 (HED/RCF 관례), train에 augmentation 자동 적용
- 평가: A = ODS/OIS/AP, B = SC/PRI/VI/F_b (BSDS 공식 방식)
- 시작: `python edge/data/unified_data.py` → `EdgeDataset`, `iter_eval`, `EdgeEvaluator`, `SegEvaluator`

## 1. 만들기

**Colab**
```python
!git clone -b JuTaein https://github.com/kyeun293/Coloring-people.git
%cd Coloring-people
!python edge/data/unified_data.py
```

**로컬:** 저장소 루트에서 `python edge/data/unified_data.py`

- 자동으로 받는 원본: BSDS500(git clone), PASCAL-Context 주석(31MB, MD5 확인), COB split 목록, VOC2010 이미지(1.3GB)
- 변환은 로컬 CPU 기준 약 7분 걸려요. 결과는 `edge/edge_work/data/unified/`에 생겨요 (약 700MB).
- 이미 받은 원본은 다시 받지 않아요.
- **Colab 팁:** Colab은 세션이 끝나면 파일이 지워져요. Google Drive를 연결하고 `EDGE_WORK`를 Drive 경로로 지정하면, 한 번 만든 데이터를 다음 세션에서 그대로 써요.
  ```python
  from google.colab import drive; drive.mount('/content/drive')
  import os; os.environ['EDGE_WORK'] = '/content/drive/MyDrive/edge_work'
  !python edge/data/unified_data.py        # 두 번째부터는 이미 있는 건 건너뜀
  ```

## 2. 학습과 평가에 바로 쓰기 (`edge_datasets.py`)

### A: edge detection 학습

```python
import sys; sys.path.insert(0, 'edge/data')
import torch
from torch.utils.data import DataLoader
from edge_datasets import EdgeDataset, rcf_loss

train = EdgeDataset('train', datasets=('bsds', 'pascal'), crop=320, norm='bgr_mean', repeat={'bsds': 5})
val   = EdgeDataset('val', datasets=('bsds',), norm='bgr_mean')      # 원본 크기 그대로 → batch_size=1

for b in DataLoader(train, batch_size=8, shuffle=True, num_workers=2):
    loss = rcf_loss(model(b['image']), b['edge'], b['valid'])         # side output 리스트도 그대로 넣으면 됨
    ...
for b in DataLoader(val, batch_size=1):
    prob = torch.sigmoid(model(b['image']))                            # 평가 GT: b['bnd'] (주석자별 경계)
```

| 반환 키 | 형태 | 내용 |
|---|---|---|
| `image` | (3,H,W) float32 | 정규화된 입력. `norm='bgr_mean'` (HED, TEED, DexiNed) / `'imagenet_rgb'` (PiDiNet) / `None` (BGR 0~1) |
| `edge` | (1,H,W) float32 | **학습 라벨**: 그 픽셀에 경계를 그은 주석자 비율 (BSDS 0, 0.2, …, 1 / PASCAL 0 또는 1) |
| `valid` | (1,H,W) float32 | 1 = 실제 이미지, 0 = crop 때 채운 빈 칸. 손실에서 빼야 해요 |
| `bnd`, `seg` | (K,H,W) | val/test에만: 주석자별 경계 GT, 구역 GT (평가용) |
| `id`, `dataset` | str | 이미지 id, `'bsds'` 또는 `'pascal'` |

- **augmentation** (train에만 자동 적용, 보고서 3-1-2):
  - 크기 변환: 0.5~1.5배 무작위
  - 좌우 뒤집기
  - 90° 단위 회전
  - 320×320 무작위 crop
  - HED/RCF 관례를 따랐어요. 회전은 90° 단위만 써요 (임의 각도는 보간 때문에 1픽셀 경계가 흐려져서).
  - 크기를 바꿀 때는 구역 지도를 바꾼 뒤 경계를 다시 그려서 **선이 항상 1픽셀**이에요.
- **라벨 방식:** `edge`는 HED/RCF 관례대로 주석자 비율이에요.
  - `rcf_loss`는 RCF(Liu et al., CVPR 2017)의 표준 손실이에요.
    - 비율 0.5 이상 = 경계, 0 = 배경, 그 사이 = 애매해서 무시
    - 경계 쪽에 가중치를 크게 줘요 (λ=1.1)
  - 다른 손실(TEED, DexiNed 원래 손실 등)을 쓰고 싶으면 `edge`와 `valid`로 직접 만들면 돼요.
- **`repeat={'bsds': 5}`:** BSDS(200장)가 PASCAL(2,499장)에 묻히지 않게 BSDS를 여러 번 넣는 옵션이에요. 필요에 맞게 조절하세요.

### B: segmentation 튜닝과 평가 (학습 없음)

```python
from edge_datasets import iter_eval
for s in iter_eval('val', 'bsds'):            # 파라미터 선정은 val, 최종은 'test'로 한 번
    s['image']   # BGR uint8 원본
    s['seg']     # (주석자,H,W) 구역 GT → SC, PRI, VI
    s['bnd']     # (주석자,H,W) 경계 GT → F_b
```

## 3. 불러오기 (저수준)

```python
import sys; sys.path.insert(0, 'edge/data')
from unified_data import list_ids, load_sample

for img_id in list_ids('pascal', 'train'):            # dataset: 'bsds' | 'pascal', split: 'train' | 'val' | 'test'
    img, seg, bnd = load_sample('pascal', 'train', img_id)
```

| 값 | 형태 | 내용 |
|---|---|---|
| `img` | (H, W, 3) uint8 | **BGR** (OpenCV 순서). RGB가 필요하면 `img[..., ::-1]` |
| `seg` | (주석자 수, H, W) int32 | 구역 번호 (1부터). **B가 쓰는 GT** |
| `bnd` | (주석자 수, H, W) bool | 1픽셀 경계. **A가 쓰는 GT** (B도 구역 경계 평가에 사용) |

PASCAL은 `.npz` 안에 `cls`(원래 클래스 번호 지도, 1~459)도 있어요. 클래스 이름은 `unified/pascal/labels.txt`에 있어요.

폴더 구조:
```
edge/edge_work/data/unified/
  bsds/   images/{train,val,test}/<id>.jpg    gt/{train,val,test}/<id>.npz
  pascal/ images/{train,val,test}/<id>.jpg    gt/{train,val,test}/<id>.npz    labels.txt
  splits/{bsds,pascal}_{train,val,test}.txt
  meta.json                                    (만든 설정, 장 수, 출처)
```

## 4. 데이터 구성

| | BSDS500 | PASCAL-Context (50%) |
|---|---|---|
| train / val / test | 200 / 100 / 200 | 2,499 / 1,304 / 1,249 |
| 이미지 크기 | 481×321 또는 321×481 | 긴 변 최대 500 |
| 주석자 수 | 4~9명 (평균 5.4) | 1명 |
| 구역 수 (주석자 1명당 평균) | 21.0 | 17.0 |
| 경계 픽셀 비율 | 1.81% | 1.70% |
| 경계의 의미 | 사람이 "다른 영역"이라고 느낀 곳 (물체 내부 포함) | **클래스가 바뀌는 곳** (물체 바깥 윤곽, 배경 구분) |

## 5. GT를 만든 규칙

**BSDS500:** 원본 그대로예요. `seg` = `groundTruth[k]['Segmentation']`, `bnd` = `groundTruth[k]['Boundaries']`이고, 모든 주석자 k를 포함해요.
- `Boundaries`는 `Segmentation`의 구역 경계에서 만든 거예요.
- 정확히는 BSDS 공식 `seg2bdry(Segmentation)`을 1픽셀로 얇게(thin) 한 것과 **2,696개 주석 전부 픽셀 단위로 똑같아요** (직접 확인).
- 그래서 A와 B는 **같은 사람 주석을 경계와 구역이라는 두 관점으로** 쓰는 거예요.

**PASCAL-Context:** 원본에는 클래스 지도(`LabelMap`) 하나만 있어서 GT 두 개를 만들었어요.
- `seg`: 같은 클래스라도 **떨어져 있으면 다른 구역**으로 나눴어요 (8-연결 연결 요소).
  - 이미지의 88%에서 같은 클래스가 여러 조각으로 나뉘어 있어요. 예: 나무 기둥 양쪽의 하늘.
  - 클래스 지도를 그대로 쓰면 떨어진 조각이 한 구역이 돼서 구역 점수가 왜곡돼요.
- `unknown`(431번, 이미지의 36%에 있고 평균 면적 3%)도 **일반 구역**으로 취급해요. 평가에서 빼지 않아요.
- `bnd`: `seg`에서 **BSDS와 똑같은 규칙**(seg2bdry + thin)으로 만든 1픽셀 경계예요. 그래서 GT와 똑같은 구역 지도는 F_b = 1이 나와요.

**PASCAL에서 이미지를 고른 과정:**

| 단계 | 장 수 | 내용 |
|---|---|---|
| ① VOC2010 trainval 이미지 | 11,321 | PASCAL VOC 2010 원본 |
| ② PASCAL-Context 주석이 있는 이미지 | 10,103 | 전체 픽셀에 클래스 주석이 있는 것만. ①과 모두 짝이 맞음 |
| ③ COB 표준 분할 | 4,998 / 2,607 / 2,498 | voc_train = 공식 train, voc_val과 voc_test = 공식 val을 반으로 나눔. 서로 겹치지 않고 합치면 ②와 같음 |
| ④ **각 split에서 50% 추출 (seed 0)** | **2,499 / 1,304 / 1,249 = 5,052** | 우리가 쓰는 데이터. 목록은 `splits/pascal_*.txt` |

- 품질이나 내용으로 걸러낸 이미지는 없어요. 비율만 줄였고, 분할 구조는 COB 그대로예요.
- 전체 데이터는 BSDS 500장 + PASCAL 5,052장 = **5,552장**이에요.

## 6. 쓸 때 주의할 점

- **test는 마지막에 한 번만** 써요. 모델 선택과 파라미터 튜닝은 val로만 해요.
- **BSDS와 PASCAL 점수는 따로 보고해요.** 경계의 의미가 달라서 숫자를 직접 비교하면 안 돼요.
- **PASCAL GT의 한계:** 물체 내부의 경계(얼굴, 옷)가 없어요. **붙어 있는 같은 클래스 물체** 사이의 경계도 없어요 (사람 두 명이 붙어 있으면 한 구역).
- **HED, PiDiNet 사전학습 가중치**는 BSDS train+val로 학습된 것이라 BSDS val 점수가 부풀려져요. TEED와 DexiNed는 BIPED로 학습됐어요.
- **A의 학습 라벨:** BSDS는 주석자별 `bnd`의 평균이에요 (HED/RCF 방식, 예: 5명 중 3명 = 0.6). PASCAL은 주석자가 1명이라 0/1이에요.
  - `EdgeDataset`의 `edge`가 이 라벨이에요. RCF도 BSDS + PASCAL-Context를 함께 학습에 썼어요.
- **사전학습 가중치에서 파인튜닝**할 때는 `norm`을 그 모델이 학습한 방식에 맞춰야 해요 (PiDiNet만 `'imagenet_rgb'`).

## 7. 평가 (`evaluation.py`)

| 담당 | 대상 | GT | 표준 지표 |
|---|---|---|---|
| A | edge 확률 맵 | `bnd` | BSDS 경계 벤치마크: **ODS F** (모델 선정 기준), OIS F, AP. NMS와 얇게 만들기 후 임계값 99개, 허용 거리 = 대각선 × 0.0075 |
| B | 구역 지도 | `seg` (+ `bnd`) | BSDS 구역 벤치마크: **Covering(SC), PRI, VI** + 구역 경계의 F_b. 알고리즘별 파라미터는 val에서 선정하고, test는 그 설정으로 한 번 평가 (ODS/OIS는 계층적 알고리즘에만 선택적으로) |

### A: edge 모델 평가
```python
from evaluation import EdgeEvaluator
ev = EdgeEvaluator()                                   # 기본: 임계값 99개, 허용 거리 0.0075, NMS 적용
for b in DataLoader(EdgeDataset('val', ('bsds',)), batch_size=1):
    prob = torch.sigmoid(model(b['image']))[0, 0].cpu().numpy()      # (H,W) 0~1
    ev.add(prob, b['bnd'][0].numpy(), img_id=b['id'][0])
print(ev.summary())     # {'ODS': .., 'OIS': .., 'AP': .., 'ODS_P', 'ODS_R', 'ODS_thresh', 'n'}
```
- 속도: 이미지 1장에 약 1.5초 (CPU). 학습 중에 자주 확인할 때는 `EdgeEvaluator(nthresh=19)`로 빠르게 보고, **보고서 숫자는 기본값(99)**으로 내요.
- `ev.per_image()`: 이미지별 최적 F, `ev.pr_curve()`: PR 곡선 (보고서 그림용)
- 입력 `prob`는 0~1 확률이어야 해요. 모델 출력이 logit이면 sigmoid를 먼저 적용하세요.

### B: segmentation 평가
```python
from evaluation import SegEvaluator
ev = SegEvaluator()
for s in iter_eval('val', 'bsds'):
    region = my_segmentation(s['image'])               # (H,W) int 구역 번호 (번호 값 자체는 상관없음)
    ev.add(region, s['seg'], s['bnd'], img_id=s['id'])
print(ev.summary())     # {'SC', 'PRI', 'VI', 'F_b', 'P_b', 'R_b', 'leakage', 'fragmentation', 'regions', 'n'}
```
- `leakage`, `fragmentation`은 색칠공부 관점의 **보조 지표**(비표준)예요.
  - Leakage = 1 − Purity: 한 구역에 여러 물체가 섞인 정도 (색이 새는 정도)
  - Fragmentation = 1 − Inverse Purity: 한 물체가 여러 구역으로 쪼개진 정도
- 파라미터를 고를 때는 val에서 여러 값으로 `SegEvaluator`를 돌려 비교하고, test는 고른 값으로 한 번만 돌려요.

### 계산 방식 (BSDS 공식 MATLAB 코드와 같게 맞춘 부분)
- **NMS:** HED/RCF/PiDiNet 평가 스크립트와 같은 Dollár `edgesNms` (r=1, s=5, m=1.01)
- **임계값:** linspace(1/100, 99/100, 99). 자른 뒤 thin으로 1픽셀화 (MATLAB `bwmorph thin`과 같은 규칙)
- **매칭:** correspondPixels와 같은 매칭 개수 (허용 거리 안에서 1:1 최대 매칭, Hopcroft–Karp)
- **주석자 여러 명:**
  - 예측 픽셀은 주석자 누구와든 매칭되면 정답이에요 (P).
  - recall은 모든 주석자의 GT 픽셀을 합쳐서 계산해요 (R).
- **ODS:** 임계값마다 전체 이미지의 개수를 **합산**해 P, R을 구하고, 인접 임계값 사이를 100단계로 보간해서 F가 최대인 지점을 골라요.
- **OIS:** 이미지마다 F가 최대인 임계값을 고른 뒤, 그 임계값에서의 개수를 **합산**해서 계산해요. 이미지별 F의 평균이 아니에요.
- **AP:** recall 0, 0.01, …, 1에서 precision을 선형 보간해 합한 뒤 × 0.01
- **SC:** 주석자 전체의 (면적 × 최대 IoU) 합 ÷ 면적 합이에요. 데이터셋 전체도 개수를 합산해요 (픽셀 가중).
- **PRI:** 주석자별 Rand Index의 평균이에요. p_ij를 쓴 정의와 수학적으로 같아요. 데이터셋 전체는 이미지 평균이에요.
- **VI:** 자연로그를 쓰고, 주석자별 평균을 내요. 데이터셋 전체는 이미지 평균이에요.
- **F_b:** 예측 구역 지도의 경계를 GT와 같은 규칙(seg2bdry + thin)으로 뽑아서, 위와 같은 방식으로 매칭해요.
- **허용 거리:** PASCAL에도 BSDS와 같은 0.0075를 써요. 논문에 따라 0.011을 쓰기도 해서, 보고서에 명시해야 해요.

### 검증한 것
- GT를 그대로 예측으로 넣으면 모든 split과 데이터셋에서 SC 1, PRI 1, VI 0, F_b 1, ODS 1이 나와요.
- 경계를 3픽셀 옮기면 매칭이 유지되고 (허용 거리 약 4.4픽셀), 8픽셀 옮기면 크게 떨어져요.
- 예측이 비어 있으면 0, 구역을 하나로 두면 F_b 0이고 Leakage가 높게 나와요.
- 최대 매칭 결과가 scipy와 같아요 (무작위 그래프 300개).
- 얇게 만들기(thin) 결과가 skimage와 픽셀 단위로 같아요.
- HED 공식 가중치를 BSDS val 20장(dev set)에 돌리면 ODS 0.755 / OIS 0.777 / AP 0.806이에요. 논문의 test 200장 값(0.788 / 0.808 / 0.840)과 비슷한 수준이에요.

## 8. 출처와 라이선스

- BSDS500: https://github.com/BIDS/BSDS500 (Arbeláez et al., TPAMI 2011). 연구용
- PASCAL-Context: https://cs.stanford.edu/~roozbeh/pascal-context/ (Mottaghi et al., CVPR 2014)
- VOC2010 이미지: http://host.robots.ox.ac.uk/pascal/VOC/voc2010/ (Flickr 이미지, 원 저작권은 각 촬영자)
- PASCAL split: COB, https://cvlsegmentation.github.io/cob/code.html
