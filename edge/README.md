# edge/: Edge Detection 기반 색칠공부판 (담당: Taein)

팀 원본 코드(`web/`, `ml/`)는 수정하지 않고, edge detection 실험만 이 폴더에서 진행합니다.

| 파일 | 내용 |
|---|---|
| `edge_coloring.ipynb` | **Colab 제출용 노트북** (`런타임 → 모두 실행`, 무료 CPU로 동작) |
| `edge_coloring.py` | 노트북 소스 (Jupytext percent 형식, `# %%` = 셀). 로컬 실행/버전 관리용 |
| `run_photo.py` | 내 사진에 Sobel / Canny / HED + 후처리 적용 → `results/`에 저장 |
| `dev_set.json` | **개발용 소규모 셋** (BSDS500 val 20장, 고정. 개선 실험은 이걸로만) |
| `make_dev_set.py` | dev set 생성 스크립트 (val을 GT 복잡도 5구간 × 4장으로 뽑음) |
| `dev_eval.py` | dev set 평가 + 실험 기록 누적 (`results/dev/experiments.csv`) |
| `nbload.py` | 스크립트들이 노트북 코드를 불러오는 공통 로더 |
| `model_table.py` | 모델별 "기본 모델 → 후처리 v1 → v2 → Ground Truth" 결과 사진 표 → `results/model_table.html` |
| `MODELS.md` | **사용한 사전학습 모델 4종 정리** (출처, 버전, 전처리, 라이선스, dev 결과) |
| `results/` | **모든 결과 그림·CSV 저장 위치** (얼굴 사진이 있어 git 제외) |
| `edge_work/` | 자동 다운로드한 BSDS500·모델 가중치 (git 제외) |

검출기: Canny, Sobel (규칙 기반) / HED, PiDiNet (BSDS 학습) / TEED, DexiNed (BIPED 학습). 모두 공개 사전학습 가중치로 추론만 해요. 자세한 내용은 [MODELS.md](MODELS.md).

## 파이프라인

```
사진 → Edge 검출 (Canny / Sobel / HED, 비교 기준: 팀 원본 k-means)
     → 후처리: 이진화 → 얇게(skeletonize) → 끊긴 선 잇기(closing)
              → 구역 분할(4-연결) → 작은 구역·선 픽셀을 가까운 구역에 병합
              → 구역 경계에서 선 다시 그리기 (항상 닫힌 선)
     → 색칠공부판 + 팔레트(k-means 대표색)
     → 평가 (BSDS500): Boundary F, Leakage(색 새는 정도), Fragmentation, 구역 수, 시간
```

## 실행

- **Colab**: `edge_coloring.ipynb` 업로드 후 모두 실행. BSDS500(약 60MB), HED 모델(약 56MB)은 자동 다운로드
- **로컬 전체 평가**: `python edge/edge_coloring.py` (`N_EVAL` 환경변수로 평가 이미지 수 변경)
- **내 사진**: `python edge/run_photo.py 사진.jpg [--method sobel canny hed]`
- 결과는 항상 `edge/results/`에 저장 (Colab에서는 `/content/results/`)

## 개선 실험 흐름

1. `edge_coloring.py`에 새 후처리 함수 작성 → `POSTPROCESS` 등록부에 `'v2': ...` 추가
2. `python edge/dev_eval.py --tag v2 --method hed --pp v1 v2 --note "무엇을 바꿨는지"`
3. 출력된 "직전 기록 대비 변화"와 `results/dev/<tag>_*.png`로 확인 → 좋아지면 유지, 아니면 되돌림
4. 모든 개선이 끝나면 노트북 4-2(test)로 **최종 평가는 한 번만**

dev set은 val에서 뽑아서 test와 겹치지 않음 (test에 맞춰 튜닝하는 것 방지).

`.py`를 고친 뒤에는 노트북을 다시 만들거나, 노트북을 직접 고쳐서 둘 중 하나만 기준으로 쓰세요.
