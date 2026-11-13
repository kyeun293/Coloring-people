/**
 * coloringBook.js — 캡쳐 이미지 → 색칠공부판 + 팔레트
 *
 * 1. 블러로 잡티 제거
 * 2. k-means로 색을 K개로 단순화 → K개 대표색이 곧 팔레트
 * 3. 3x3 다수결 필터로 노이즈 정리
 * 4. 같은 색끼리 붙은 덩어리 = 구역. 너무 작은 구역은 이웃 색으로 흡수
 * 5. 구역 경계만 검은 선, 나머지 흰색
 *
 * 경계선이 구역에서 나오므로 항상 닫혀 있음 → 칠할 때 색이 새지 않음
 */

export function buildColoringBook(imageData, { K = 7, minAreaRatio = 0.002 } = {}) {
  const { width: w, height: h } = imageData;
  const N = w * h;

  const rgb = boxBlur(imageData.data, w, h, 2);
  const { centers, labels } = kmeans(rgb, N, K);

  modeFilter(labels, w, h, K);
  modeFilter(labels, w, h, K);

  const minArea = Math.max(40, Math.floor(N * minAreaRatio));
  for (let pass = 0; pass < 5; pass++) {
    if (!absorbSmallRegions(labels, w, h, K, minArea)) break;
  }

  const { map: regionMap, sizes, count } = components(labels, w, h);

  // 구역별 픽셀 목록 (채우기를 빠르게)
  const regionStart = new Int32Array(count + 1);
  for (let r = 0; r < count; r++) regionStart[r + 1] = regionStart[r] + sizes[r];
  const regionPixels = new Int32Array(N);
  const cursor = regionStart.slice(0, count);
  for (let i = 0; i < N; i++) regionPixels[cursor[regionMap[i]]++] = i;

  // 경계선: 오른쪽/아래 픽셀과 구역이 다르면 선
  const line = new Uint8Array(N);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, r = regionMap[i];
      if ((x + 1 < w && regionMap[i + 1] !== r) || (y + 1 < h && regionMap[i + w] !== r)) line[i] = 1;
    }
  }

  // 팔레트: 실제로 쓰인 색만, 많이 쓰인 순
  const usage = new Array(K).fill(0);
  for (let i = 0; i < N; i++) usage[labels[i]]++;
  const palette = [...Array(K).keys()]
    .filter(k => usage[k] > 0)
    .sort((a, b) => usage[b] - usage[a])
    .map(k => [0, 1, 2].map(c => Math.round(centers[k * 3 + c])));

  return new ColoringBoard(w, h, regionMap, regionStart, regionPixels, line, palette);
}

/** 색칠판 상태 + 그리기용 캔버스 */
export class ColoringBoard {
  constructor(w, h, regionMap, regionStart, regionPixels, line, palette) {
    Object.assign(this, { w, h, regionMap, regionStart, regionPixels, line, palette });
    this.canvas = document.createElement('canvas');
    this.canvas.width = w; this.canvas.height = h;
    this.ctx = this.canvas.getContext('2d');
    this.image = this.ctx.createImageData(w, h);
    const d = this.image.data;
    for (let i = 0; i < w * h; i++) {
      const v = line[i] ? 35 : 255;
      d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v;
      d[i * 4 + 3] = 255;
    }
    this.ctx.putImageData(this.image, 0, 0);
  }

  /** u, v: 색칠판 안의 상대 위치 (0~1) */
  fillAt(u, v, [r, g, b]) {
    if (u < 0 || u >= 1 || v < 0 || v >= 1) return false;
    const id = this.regionMap[Math.floor(v * this.h) * this.w + Math.floor(u * this.w)];
    const d = this.image.data;
    for (let p = this.regionStart[id]; p < this.regionStart[id + 1]; p++) {
      const i = this.regionPixels[p];
      if (this.line[i]) continue;
      d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b;
    }
    this.ctx.putImageData(this.image, 0, 0);
    return true;
  }
}

// ── 내부 처리 ────────────────────────────────────────

/** RGBA → 블러된 RGB(Float32). 가로·세로 박스 블러 반복 */
function boxBlur(src, w, h, passes) {
  const N = w * h;
  let a = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) { a[i*3] = src[i*4]; a[i*3+1] = src[i*4+1]; a[i*3+2] = src[i*4+2]; }
  let b = new Float32Array(N * 3);
  for (let p = 0; p < passes; p++) {
    for (const [dx, dy] of [[1, 0], [0, 1]]) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const x0 = Math.max(0, x - dx), x1 = Math.min(w - 1, x + dx);
          const y0 = Math.max(0, y - dy), y1 = Math.min(h - 1, y + dy);
          const i = (y * w + x) * 3, i0 = (y0 * w + x0) * 3, i1 = (y1 * w + x1) * 3;
          for (let c = 0; c < 3; c++) b[i + c] = (a[i0 + c] + a[i + c] + a[i1 + c]) / 3;
        }
      }
      [a, b] = [b, a];
    }
  }
  return a;
}

/** 샘플 픽셀로 k-means++ 학습 후 전체 픽셀 할당 */
function kmeans(rgb, N, K, iters = 12) {
  const S = Math.min(5000, N), step = N / S;
  const sample = new Float32Array(S * 3);
  for (let s = 0; s < S; s++) {
    const p = Math.floor(s * step) * 3;
    sample[s*3] = rgb[p]; sample[s*3+1] = rgb[p+1]; sample[s*3+2] = rgb[p+2];
  }
  const d2 = (arr, i, C, k) => {
    const dr = arr[i] - C[k*3], dg = arr[i+1] - C[k*3+1], db = arr[i+2] - C[k*3+2];
    return dr*dr + dg*dg + db*db;
  };

  // k-means++ 초기화: 서로 멀리 떨어진 색을 초기 중심으로
  const C = new Float32Array(K * 3);
  const first = Math.floor(S / 2);
  C.set(sample.subarray(first * 3, first * 3 + 3), 0);
  const D = new Float32Array(S).fill(Infinity);
  for (let k = 1; k < K; k++) {
    let total = 0;
    for (let s = 0; s < S; s++) { D[s] = Math.min(D[s], d2(sample, s*3, C, k-1)); total += D[s]; }
    let pick = Math.floor(Math.random() * S);
    if (total > 0) {
      let t = Math.random() * total;
      for (let s = 0; s < S; s++) { t -= D[s]; if (t <= 0) { pick = s; break; } }
    }
    C.set(sample.subarray(pick * 3, pick * 3 + 3), k * 3);
  }

  const nearest = (arr, i) => {
    let best = 0, bd = Infinity;
    for (let k = 0; k < K; k++) { const d = d2(arr, i, C, k); if (d < bd) { bd = d; best = k; } }
    return best;
  };

  for (let it = 0; it < iters; it++) {
    const sum = new Float64Array(K * 3), cnt = new Uint32Array(K);
    for (let s = 0; s < S; s++) {
      const k = nearest(sample, s * 3);
      sum[k*3] += sample[s*3]; sum[k*3+1] += sample[s*3+1]; sum[k*3+2] += sample[s*3+2]; cnt[k]++;
    }
    for (let k = 0; k < K; k++) {
      if (!cnt[k]) continue;
      C[k*3] = sum[k*3] / cnt[k]; C[k*3+1] = sum[k*3+1] / cnt[k]; C[k*3+2] = sum[k*3+2] / cnt[k];
    }
  }

  const labels = new Uint8Array(N);
  for (let i = 0; i < N; i++) labels[i] = nearest(rgb, i * 3);
  return { centers: C, labels };
}

/** 3x3 다수결: 각 픽셀을 주변에서 가장 많은 색으로 */
function modeFilter(labels, w, h, K) {
  const out = labels.slice();
  const cnt = new Uint8Array(K);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      cnt.fill(0);
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) cnt[labels[(y+dy)*w + x+dx]]++;
      let best = labels[y*w + x];
      for (let k = 0; k < K; k++) if (cnt[k] > cnt[best]) best = k;
      out[y*w + x] = best;
    }
  }
  labels.set(out);
}

/** 4방향 연결 요소 라벨링 */
function components(labels, w, h) {
  const N = w * h;
  const map = new Int32Array(N).fill(-1);
  const stack = new Int32Array(N);
  const sizes = [];
  let count = 0;
  for (let i = 0; i < N; i++) {
    if (map[i] !== -1) continue;
    const l = labels[i];
    let sp = 0, size = 0;
    stack[sp++] = i; map[i] = count;
    while (sp) {
      const p = stack[--sp]; size++;
      const x = p % w;
      if (x > 0     && map[p-1] === -1 && labels[p-1] === l) { map[p-1] = count; stack[sp++] = p-1; }
      if (x < w - 1 && map[p+1] === -1 && labels[p+1] === l) { map[p+1] = count; stack[sp++] = p+1; }
      if (p >= w    && map[p-w] === -1 && labels[p-w] === l) { map[p-w] = count; stack[sp++] = p-w; }
      if (p + w < N && map[p+w] === -1 && labels[p+w] === l) { map[p+w] = count; stack[sp++] = p+w; }
    }
    sizes.push(size);
    count++;
  }
  return { map, sizes, count };
}

/** 작은 구역을 가장 많이 맞닿은 이웃 색으로 바꿈. 바뀐 게 있으면 true */
function absorbSmallRegions(labels, w, h, K, minArea) {
  const { map, sizes, count } = components(labels, w, h);
  const small = sizes.map(s => s < minArea);
  if (!small.some(Boolean)) return false;

  const votes = new Uint32Array(count * K);
  const N = w * h;
  for (let i = 0; i < N; i++) {
    const r = map[i];
    if (!small[r]) continue;
    const x = i % w;
    if (x > 0     && map[i-1] !== r) votes[r*K + labels[i-1]]++;
    if (x < w - 1 && map[i+1] !== r) votes[r*K + labels[i+1]]++;
    if (i >= w    && map[i-w] !== r) votes[r*K + labels[i-w]]++;
    if (i + w < N && map[i+w] !== r) votes[r*K + labels[i+w]]++;
  }
  const target = new Int16Array(count).fill(-1);
  for (let r = 0; r < count; r++) {
    if (!small[r]) continue;
    let best = -1, bv = 0;
    for (let k = 0; k < K; k++) if (votes[r*K + k] > bv) { bv = votes[r*K + k]; best = k; }
    target[r] = best;
  }
  let changed = false;
  for (let i = 0; i < N; i++) {
    const t = target[map[i]];
    if (t >= 0 && labels[i] !== t) { labels[i] = t; changed = true; }
  }
  return changed;
}
