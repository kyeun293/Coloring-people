# edge/: Edge Detection 기반 색칠공부판 (담당: Taein)

팀 원본 코드(`web/`, `ml/`)는 수정하지 않고, edge detection 실험만 이 폴더에서 진행합니다.

| 파일 | 내용 |
|---|---|
| `edge_coloring.ipynb` | **Colab 제출용 노트북** (`런타임 → 모두 실행`, 무료 CPU로 동작) |
| `edge_coloring.py` | 노트북 소스 (Jupytext percent 형식, `# %%` = 셀). 로컬 실행/버전 관리용 |

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
- **로컬**: `python edge/edge_coloring.py` (데이터는 `edge_work/`에 저장. `EDGE_WORK`, `N_EVAL` 환경변수로 변경)

`.py`를 고친 뒤에는 노트북을 다시 만들거나, 노트북을 직접 고쳐서 둘 중 하나만 기준으로 쓰세요.
