/**
 * edgeModel.js — 학습된 엣지 모델(TEED, ml/train_edge.py) 웹 추론
 *
 * 입력: ImageData (가로·세로 8의 배수 권장 — 480x360)
 * 출력: Float32Array(w*h), 각 픽셀이 경계일 확률 0~1
 * 모델이 없거나 실패하면 null → coloringBook은 기존 색 기반 방식으로만 동작
 */

const MODEL_URL = '/models/edge.onnx';
let sessionPromise = null;

function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(MODEL_URL)
      .catch(err => { console.warn('[edge] 모델 로드 실패, 색 기반으로만 진행', err); return null; });
  }
  return sessionPromise;
}

/** 앱 시작 시 미리 불러두기 (첫 촬영 지연 줄이기) */
export function preloadEdgeModel() { getSession(); }

export async function detectEdges(imageData) {
  const session = await getSession();
  if (!session) return null;
  const { width: w, height: h, data } = imageData;
  const N = w * h;
  const input = new Float32Array(3 * N);   // NCHW, RGB 0~255 (정규화는 모델 안에서)
  for (let i = 0; i < N; i++) {
    input[i] = data[i * 4]; input[N + i] = data[i * 4 + 1]; input[2 * N + i] = data[i * 4 + 2];
  }
  try {
    const out = await session.run({ rgb: new ort.Tensor('float32', input, [1, 3, h, w]) });
    return out.edge.data;
  } catch (err) {
    console.warn('[edge] 추론 실패, 색 기반으로만 진행', err);
    return null;
  }
}
