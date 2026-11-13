/**
 * gestures.js — 학습 없이 랜드마크 규칙으로 손 모양 판별
 *
 * MediaPipe 좌표(x, y)는 0~1로 정규화돼 있어서 가로세로 비율이 다름 →
 * 거리 계산 시 x에 영상 비율(aspect = videoWidth / videoHeight)을 곱해 보정.
 */

const dist = (a, b, aspect) => Math.hypot((a.x - b.x) * aspect, a.y - b.y);

// [손가락 끝, PIP 관절] — 검지, 중지, 약지, 새끼
const FINGERS = { index: [8, 6], middle: [12, 10], ring: [16, 14], pinky: [20, 18] };

// 손목→끝 거리가 손목→PIP 거리보다 확실히 길면 펴진 것
function extended(hand, finger, aspect) {
  const [tip, pip] = FINGERS[finger];
  return dist(hand[0], hand[tip], aspect) > dist(hand[0], hand[pip], aspect) * 1.1;
}

// 손목→끝 거리가 손목→PIP 거리보다 짧으면 접힌 것
function folded(hand, finger, aspect) {
  const [tip, pip] = FINGERS[finger];
  return dist(hand[0], hand[tip], aspect) < dist(hand[0], hand[pip], aspect);
}

/** 엄지 끝–검지 끝 거리 / 손 크기(손목–중지 MCP). 카메라와의 거리에 무관 */
export function pinchRatio(hand, aspect) {
  const size = dist(hand[0], hand[9], aspect) || 1e-6;
  return dist(hand[4], hand[8], aspect) / size;
}

/** ✌️ 검지·중지 펴고 약지·새끼 접음 (엄지는 무시) */
export function isVSign(hand, aspect) {
  return extended(hand, 'index', aspect) && extended(hand, 'middle', aspect)
      && folded(hand, 'ring', aspect)    && folded(hand, 'pinky', aspect);
}

/** ✊ 네 손가락 모두 접음. 집기(엄지·검지 붙임)와 헷갈리지 않게 집기 상태는 제외 */
export function isFist(hand, aspect) {
  return folded(hand, 'index', aspect) && folded(hand, 'middle', aspect)
      && folded(hand, 'ring', aspect)  && folded(hand, 'pinky', aspect)
      && pinchRatio(hand, aspect) > 0.35;
}

/**
 * 집기 판별. 집는 기준(0.28)과 놓는 기준(0.42)을 다르게 둬서
 * 경계값 근처에서 집었다 놨다 떨리는 것을 막음.
 */
export class PinchDetector {
  constructor() { this.on = false; }
  reset() { this.on = false; }
  update(hand, aspect) {
    const prev = this.on;
    if (!hand) {
      this.on = false;
    } else {
      const r = pinchRatio(hand, aspect);
      if (!this.on && r < 0.28) this.on = true;
      else if (this.on && r > 0.42) this.on = false;
    }
    return { on: this.on, started: !prev && this.on, ended: prev && !this.on };
  }
}

/**
 * 같은 손 모양을 일정 시간 유지해야 발동. progress(0~1)로 진행도 표시.
 */
export class HoldTrigger {
  constructor(ms) { this.ms = ms; this.since = null; }
  reset() { this.since = null; }
  update(active, now) {
    if (!active) { this.since = null; return { progress: 0, fired: false }; }
    if (this.since === null) this.since = now;
    const progress = Math.min(1, (now - this.since) / this.ms);
    if (progress >= 1) { this.since = null; return { progress: 1, fired: true }; }
    return { progress, fired: false };
  }
}

/** object-fit: cover 로 표시된 영상이 화면에서 차지하는 영역 */
export function videoRect(video, W, H) {
  const scale = Math.max(W / video.videoWidth, H / video.videoHeight);
  const w = video.videoWidth * scale, h = video.videoHeight * scale;
  return { x: (W - w) / 2, y: (H - h) / 2, w, h };
}

/** 랜드마크 → 화면 좌표 (영상이 좌우 반전돼 있으므로 x 반전) */
export function toScreen(lm, rect) {
  return { x: (1 - lm.x) * rect.w + rect.x, y: lm.y * rect.h + rect.y };
}
