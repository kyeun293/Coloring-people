# 통일 데이터셋: BSDS500 + PASCAL-Context (담당: Taein)

edge detection(A)과 segmentation(B)이 **같은 데이터, 같은 형식, 같은 split**을 쓰도록 정리한 폴더예요.
데이터 파일 자체는 용량(약 700MB)과 라이선스 때문에 git에 넣지 않았어요. 스크립트를 실행하면 원래 배포처에서 받아서 **모두에게 똑같은 데이터**가 만들어져요.

| 파일 | 내용 |
|---|---|
| `unified_data.py` | 다운로드 → 변환 스크립트 + 불러오기 함수 (`list_ids`, `load_sample`) |
| `splits/*.txt` | **팀 공용 고정 split 목록** (어떤 이미지를 train/val/test로 쓰는지). 이 목록을 바꾸지 마세요 |

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

## 2. 불러오기

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

## 3. 데이터 구성

| | BSDS500 | PASCAL-Context (50%) |
|---|---|---|
| train / val / test | 200 / 100 / 200 | 2,499 / 1,304 / 1,249 |
| 이미지 크기 | 481×321 또는 321×481 | 긴 변 최대 500 |
| 주석자 수 | 4~9명 (평균 5.4) | 1명 |
| 구역 수 (주석자 1명당 평균) | 21.0 | 17.0 |
| 경계 픽셀 비율 | 1.81% | 1.70% |
| 경계의 의미 | 사람이 "다른 영역"이라고 느낀 곳 (물체 내부 포함) | **클래스가 바뀌는 곳** (물체 바깥 윤곽, 배경 구분) |

## 4. GT를 만든 규칙

**BSDS500:** 원본 그대로예요. `seg` = `groundTruth[k]['Segmentation']`, `bnd` = `groundTruth[k]['Boundaries']`이고, 모든 주석자 k를 포함해요.
- `Boundaries`는 `Segmentation`의 구역 경계에서 만든 거예요.
- 2,696개 주석 전부에서 경계 픽셀이 라벨이 바뀌는 곳의 1픽셀 이내에 있는 걸 확인했어요.
- 그래서 A와 B는 **같은 사람 주석을 경계와 구역이라는 두 관점으로** 쓰는 거예요.

**PASCAL-Context:** 원본에는 클래스 지도(`LabelMap`) 하나만 있어서 GT 두 개를 만들었어요.
- `seg`: 같은 클래스라도 **떨어져 있으면 다른 구역**으로 나눴어요 (8-연결 연결 요소).
  - 이미지의 88%에서 같은 클래스가 여러 조각으로 나뉘어 있어요. 예: 나무 기둥 양쪽의 하늘.
  - 클래스 지도를 그대로 쓰면 떨어진 조각이 한 구역이 돼서 구역 점수가 왜곡돼요.
- `unknown`(431번, 이미지의 36%에 있고 평균 면적 3%)도 **일반 구역**으로 취급해요. 평가에서 빼지 않아요.
- `bnd`: `seg`에서 BSDS의 `seg2bdry`와 같은 방식으로 만든 1픽셀 경계예요.

**PASCAL split:** COB 논문(Maninis et al., ECCV 2016)의 표준 분할에서 **각각 50%**를 고정 seed(0)로 뽑았어요.
- 원래 분할은 voc_train 4,998 / voc_val 2,607 / voc_test 2,498장이에요. voc_val과 voc_test는 공식 PASCAL-Context val을 반으로 나눈 거예요.
- 뽑은 목록은 `splits/`에 고정해 뒀어요.

## 5. 쓸 때 주의할 점

- **test는 마지막에 한 번만** 써요. 모델 선택과 파라미터 튜닝은 val로만 해요.
- **BSDS와 PASCAL 점수는 따로 보고해요.** 경계의 의미가 달라서 숫자를 직접 비교하면 안 돼요.
- **PASCAL GT의 한계:** 물체 내부의 경계(얼굴, 옷)가 없어요. **붙어 있는 같은 클래스 물체** 사이의 경계도 없어요 (사람 두 명이 붙어 있으면 한 구역).
- **HED, PiDiNet 사전학습 가중치**는 BSDS train+val로 학습된 것이라 BSDS val 점수가 부풀려져요. TEED와 DexiNed는 BIPED로 학습됐어요.
- **A의 학습 라벨:** BSDS는 주석자별 `bnd`를 평균내서 만드는 게 관례예요 (HED/RCF 방식, 예: 5명 중 3명 = 0.6). PASCAL은 주석자가 1명이라 0/1이에요.
  - RCF도 BSDS + PASCAL-Context를 함께 학습에 썼어요.

## 6. 평가 기준 (요약, 평가 코드는 따로 제공 예정)

| 담당 | 대상 | GT | 표준 지표 |
|---|---|---|---|
| A | edge 확률 맵 | `bnd` | BSDS 경계 벤치마크: **ODS F** (모델 선정 기준), OIS F, AP. NMS와 얇게 만들기 후 임계값 99개, 허용 거리 = 대각선 × 0.0075 |
| B | 구역 지도 | `seg` (+ `bnd`) | BSDS 구역 벤치마크: **Covering(SC), PRI, VI** + 구역 경계의 F_b. 파라미터 하나를 scale로 바꿔 가며 ODS/OIS 산출 |

## 7. 출처와 라이선스

- BSDS500: https://github.com/BIDS/BSDS500 (Arbeláez et al., TPAMI 2011). 연구용
- PASCAL-Context: https://cs.stanford.edu/~roozbeh/pascal-context/ (Mottaghi et al., CVPR 2014)
- VOC2010 이미지: http://host.robots.ox.ac.uk/pascal/VOC/voc2010/ (Flickr 이미지, 원 저작권은 각 촬영자)
- PASCAL split: COB, https://cvlsegmentation.github.io/cob/code.html
