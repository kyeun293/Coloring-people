# coloring-people 🖐️🎨

웹캠 앞에서 손 모션으로 화면을 캡쳐하면, 그 장면이 **색칠공부판**으로 바뀌고
손가락으로 팔레트의 색을 집어 원하는 구역에 칠할 수 있는 브라우저 앱입니다.

MediaPipe로 손 랜드마크를 실시간 추적하고, 캡쳐한 이미지를 색 단순화 + 구역 분할로
색칠공부판과 팔레트로 만듭니다. 서버나 GPU 없이 브라우저에서 전부 동작합니다.

> 🚧 **개발 중** — 손 추적, 제스처 분류, 데이터 수집/학습 파이프라인은 구현되어 있고,
> 색칠공부 기능(캡쳐 → 색칠판 → 팔레트 → 채우기)은 구현 예정입니다.

## 동작 흐름

1. **캡쳐** — 시작 모션을 하면 3초 카운트다운 후 화면의 일정 영역을 캡쳐
2. **색칠공부판 생성** — 캡쳐 이미지의 선을 따고 배경을 흰색으로 바꿔 색칠공부판을 만들고, 캠 화면을 가리도록 계속 띄워 둠
3. **팔레트** — 캡쳐한 장면의 주요 색상을 뽑아 화면 상단에 팔레트로 표시
4. **색칠** — 팔레트 위에서 엄지·검지를 맞붙여 색을 집고, 색칠판 위에서 손가락을 떼면 그 구역이 해당 색으로 채워짐
5. **종료** — 종료 모션을 하면 색칠 모드 종료

## 구현 방식

| 기능 | 방법 |
|---|---|
| 시작·종료 모션 | 학습된 제스처 분류기 또는 규칙 기반 판별 (모션 종류는 미정) |
| 캡쳐 | 카운트다운 후 비디오 프레임의 일정 영역을 캔버스로 잘라냄 |
| 색칠판 + 팔레트 | 블러 → **k-means로 색을 K개(6~8개)로 단순화** → 같은 색끼리 붙은 영역을 구역으로 나누고 작은 구역은 이웃에 병합 → 구역 경계만 검은 선, 나머지는 흰색 |
| 팔레트 색상 | k-means로 나온 K개 대표색을 그대로 사용 |
| 색 집기 | 엄지 끝(랜드마크 4)과 검지 끝(8) 사이 거리가 임계값 이하면 "집음" |
| 채우기 | 손을 뗀 위치가 속한 구역 번호를 찾아 그 구역 전체를 선택한 색으로 칠함 |

> 💡 Canny 같은 윤곽선 검출만 쓰면 선이 중간중간 끊겨서, 칠할 때 색이 새어 나갑니다.
> 구역 분할로 만든 경계선은 항상 닫혀 있어서 색이 샐 일이 없습니다.

## 기술 스택

- **프론트엔드**: Vanilla JS (ES Modules), Canvas 2D
- **손 인식**: [MediaPipe Tasks Vision](https://developers.google.com/mediapipe) (HandLandmarker)
- **제스처 추론**: [onnxruntime-web](https://onnxruntime.ai/)
- **학습**: PyTorch, scikit-learn, ONNX export

## 프로젝트 구조

```
.
├── web/                # 프론트엔드 (정적 서버로 실행)
│   ├── index.html
│   ├── models/         # handseal.onnx, labels.json (학습 후 자동 복사)
│   └── src/
│       ├── main.js                 # 카메라, MediaPipe, 메인 루프
│       ├── core/
│       │   ├── gestureClassifier.js  # ONNX 제스처 분류
│       │   └── dataCollector.js      # 학습 데이터 녹화/CSV 저장
│       ├── effects/
│       │   └── arEffectEngine.js     # 캔버스 렌더링 (랜드마크, 이펙트)
│       └── ui/
│           ├── uiController.js
│           └── style.css
└── ml/
    ├── train.py        # CSV → 정규화 → 증강 → MLP 학습 → ONNX 변환
    ├── data/raw/       # 수집한 CSV
    └── models/
```

## 실행 방법

### 1. 웹앱 실행

`web/` 폴더를 루트로 하는 정적 서버로 띄웁니다 (모델을 `/models/handseal.onnx` 경로로 불러오기 때문에 반드시 `web/` 기준).

```bash
cd web
python -m http.server 5500
```

브라우저에서 http://localhost:5500 접속 후 카메라 권한을 허용하세요.
VS Code Live Server를 쓸 경우 `web/` 폴더를 루트로 열어야 합니다.

> 카메라가 다른 앱이나 탭에서 사용 중이면 `NotReadableError`가 납니다. 다른 탭과 앱을 닫고 새로고침하세요.

### 2. 제스처 학습 데이터 수집

1. 하단 모드 바에서 **DATA COLLECT** 선택
2. 레이블 버튼으로 제스처 선택
3. 포즈를 잡고 **녹화 시작** → 카운트다운 후 자동 녹화
4. 제스처당 500프레임 이상 권장 (`↩ 마지막 녹화 취소`로 실수 삭제 가능)
5. **💾 CSV 저장** → 파일을 `ml/data/raw/`로 이동

### 3. 모델 학습

```bash
pip install torch scikit-learn pandas numpy onnx
python ml/train.py
```

학습이 끝나면 `ml/models/handseal.onnx`가 생성되고 `web/models/`로 자동 복사됩니다. 새로고침하면 바로 반영됩니다.
모델이 없으면 규칙 기반 fallback으로 동작합니다.
