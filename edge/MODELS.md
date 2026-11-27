# 사용한 사전학습 Edge 모델 4종

학습 없이 공개 가중치로 **추론만** 했어요. 가중치 파일은 저장소에 넣지 않았고, 노트북(`edge_coloring.ipynb`)을 실행하면 아래 출처에서 자동으로 받아요.
(DexiNed 가중치는 141MB라 GitHub 파일 제한 100MB를 넘어요. 다른 모델도 원 저자 배포처를 그대로 쓰는 게 라이선스상 안전해요.)

## 한눈에 보기

| 모델 | 발표 | 파라미터 | 가중치 파일 (크기) | 학습 데이터 | 라이선스 |
|---|---|---|---|---|---|
| **HED** | ICCV 2015 | 약 14.7M | `hed_pretrained_bsds.caffemodel` (59MB) | BSDS500 train+val | 저장소에 별도 표기 없음 (Caffe 기반, 연구용으로 사용) |
| **PiDiNet** | ICCV 2021 | 0.71M | `table5_pidinet.pth` (2.9MB) | BSDS500 train+val (+PASCAL VOC) | 연구용 (상업적 사용은 저자 문의) |
| **TEED** | ICCV 2023 Workshop | 0.059M | `checkpoints/BIPED/5/5_model.pth` (0.25MB) | BIPED | MIT |
| **DexiNed** | WACV 2020 | 35.2M | `DexiNed_BIPED_10.pth` (141MB) | BIPED | MIT (kornia 배포) |

## 모델별 상세

### HED (Holistically-Nested Edge Detection)
- 논문: Xie & Tu, "Holistically-Nested Edge Detection", ICCV 2015
- 구조: VGG16의 5단계 출력(side output)을 합쳐(fuse) edge 확률을 만들어요.
- 출처:
  - 구조 파일: `https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt` (원 저자 저장소)
  - 가중치: `https://raw.githubusercontent.com/ashukid/hed-edge-detector/master/hed_pretrained_bsds.caffemodel` (원 저자 BSDS 가중치의 미러)
- 실행: OpenCV `cv2.dnn` (Caffe 형식). PyTorch 없이 CPU로 돌아가요. Crop 레이어는 직접 등록해요 (`CropLayer`).
- 입력: BGR, 평균 `(104.007, 116.669, 122.679)`를 빼요. 출력은 sigmoid가 적용된 fuse 맵이에요.

### PiDiNet (Pixel Difference Networks)
- 논문: Su et al., "Pixel Difference Networks for Efficient Edge Detection", ICCV 2021
- 구조: 픽셀 차이 합성곱(PDC)으로 Sobel 같은 기울기 연산을 학습형으로 넣은 경량 모델이에요.
- 출처: `https://github.com/hellozhuo/pidinet` (확인한 커밋 `d21aa88`, 2024-07-02)
- 가중치: 저장소의 `trained_models/table5_pidinet.pth` (논문 Table 5 기본 모델)
- 설정: `config=carv4, dil=True, sa=True` (논문 기본값)
- 입력: RGB, ImageNet 평균/표준편차로 정규화해요. 크기는 16의 배수로 맞춰요. 출력은 마지막 fuse 맵이에요 (sigmoid 적용됨).

### TEED (Tiny and Efficient Edge Detector)
- 논문: Soria et al., "Tiny and Efficient Model for the Edge Detection Generalization", ICCV 2023 Workshop (arXiv 2308.06468)
- 구조: 파라미터 5.9만 개인 초경량 모델이에요. DexiNed 저자가 만들었어요.
- 출처: `https://github.com/xavysp/TEED` (확인한 커밋 `40fa4b1`, 2023-10-05)
- 가중치: 저장소의 `checkpoints/BIPED/5/5_model.pth` (BIPED 5 epoch, 249,139 바이트)
- 입력: BGR에서 평균 `(103.939, 116.779, 123.68)`을 빼요. 짧은 변이 512보다 작으면 1.5배 키워서 추론해요 (원 저자 테스트 방식).
- 출력: fuse logit → sigmoid → 이미지별 최소~최대 정규화 (원 저자 코드와 같음).
- 참고: 저장소 `main.py`에 "BIPED+BRIND" 언급이 있어요. BRIND는 BSDS 이미지로 만든 데이터셋이라, BSDS를 본 적이 있는지는 논문으로 확인이 필요해요.

### DexiNed (Dense Extreme Inception Network)
- 논문: Soria et al., "Dense Extreme Inception Network: Towards a Robust CNN Model for Edge Detection", WACV 2020
- 출처: `kornia` 라이브러리 내장 (`kornia.models.dexined.DexiNed(pretrained=True)`, 확인한 버전 kornia 0.9.2). 가중치는 처음 실행할 때 `~/.cache/torch/hub/checkpoints/`로 자동 다운로드돼요.
- 입력: BGR에서 평균 `(103.939, 116.779, 123.68)`을 빼요. 크기는 16의 배수로 맞춰요.
- 출력: fuse logit → sigmoid → 이미지별 최소~최대 정규화.

## dev set 결과 (BSDS500 val 20장, `dev_eval.py --tag models`)

후처리 파라미터는 튜닝 없이 모두 HED 기본값(thresh=0.3, closing 3, 최소 구역 0.2%)을 썼어요.

| 모델 | 후처리 | Region F (주 지표) | Boundary F | Leakage | Fragmentation | 구역 수 |
|---|---|---|---|---|---|---|
| 팀 원본 k-means | – | 0.535 | 0.554 | 0.139 | 0.594 | 53.3 |
| Canny | v1 | 0.640 | 0.176 | 0.466 | 0.104 | 10.3 |
| Sobel | v1 | 0.647 | 0.178 | 0.501 | 0.036 | 5.2 |
| **HED** ⚠️ | v1 | **0.794** | **0.661** | 0.197 | 0.192 | 18.6 |
| **PiDiNet** ⚠️ | v1 | 0.758 | 0.655 | 0.222 | 0.226 | 24.0 |
| **TEED** | v1 | 0.666 | 0.246 | 0.461 | 0.058 | 7.8 |
| **DexiNed** | v1 | 0.738 | 0.514 | 0.276 | 0.205 | 22.7 |

⚠️ HED와 PiDiNet은 BSDS500 train+val로 학습돼서 val(dev set) 점수가 실제보다 좋게 나올 수 있어요. 공정한 비교는 마지막에 test로 한 번만 해요.

후처리 v2(NMS + hysteresis)는 모든 모델에서 Region F가 떨어져서 기각했어요. 지표 정의와 채택 규칙은 노트북 4장에 있어요.

## 재현 방법

```bash
python edge/dev_eval.py --tag models --pp v1 v2      # 위 표 숫자
python edge/model_table.py                           # 모델별 결과 사진 표 → edge/results/model_table.html
```
