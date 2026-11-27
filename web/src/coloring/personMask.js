/**
 * personMask.js — 캡쳐 이미지에서 사람 영역 구하기 (MediaPipe Selfie Segmenter)
 *
 * 출력: Float32Array(w*h), 각 픽셀이 사람일 확률 0~1
 * 실패하거나 결과가 이상하면 null → 사람 마스크 없이 진행
 */

import { FilesetResolver, ImageSegmenter }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/vision_bundle.js';

const WASM_URL  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite';

let segmenterPromise = null;

function getSegmenter() {
  if (!segmenterPromise) {
    segmenterPromise = FilesetResolver.forVisionTasks(WASM_URL)
      .then(vision => ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL },
        runningMode: 'IMAGE',
        outputConfidenceMasks: true,
        outputCategoryMask: false,
      }))
      .catch(err => { console.warn('[person] 사람 분할 모델 로드 실패, 마스크 없이 진행', err); return null; });
  }
  return segmenterPromise;
}

export function preloadPersonMask() { getSegmenter(); }

/** imageData와 같은 크기의 캔버스를 받아 사람 확률 맵을 돌려줌 */
export async function detectPerson(canvas) {
  const segmenter = await getSegmenter();
  if (!segmenter) return null;
  const { width: w, height: h } = canvas;
  let mask = null;
  try {
    // 콜백 안에서만 마스크가 유효하므로 그 자리에서 복사
    segmenter.segment(canvas, result => {
      const m = result.confidenceMasks?.[0];
      if (!m) return;
      const arr = m.getAsFloat32Array ? m.getAsFloat32Array() : m;
      if (arr.length === w * h) mask = Float32Array.from(arr);
    });
  } catch (err) {
    console.warn('[person] 사람 분할 실패, 마스크 없이 진행', err);
    return null;
  }
  if (!mask) return null;

  // 사람이 거의 없거나 화면 전체가 사람으로 나오면 쓸모없으니 버림
  let sum = 0;
  for (let i = 0; i < mask.length; i++) sum += mask[i] > 0.5 ? 1 : 0;
  const ratio = sum / mask.length;
  if (ratio < 0.02 || ratio > 0.95) {
    console.log(`[person] 사람 영역 ${Math.round(ratio * 100)}% → 마스크 사용 안 함`);
    return null;
  }
  return mask;
}
