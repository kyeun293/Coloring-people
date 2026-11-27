/**
 * coloringBook.js — 캡쳐 이미지 → 색칠공부판 + 팔레트
 *
 * 1. 블러로 잡티 제거
 * 2. k-means로 색을 K개로 단순화 → K개 대표색이 곧 팔레트
 * 3. 3x3 다수결 필터로 노이즈 정리
 * 4. 구역 나누기: 같은 색끼리 붙은 덩어리 = 구역
 *    (사람 마스크(person)가 있으면 먼저: 배경의 엣지는 약하게, 사람 외곽선은 확실한 경계로)
 *    + 학습된 엣지 모델(edges)이 있으면
 *      a. 모델이 "확실한 경계"라고 한 곳은 같은 색이어도 구역을 자름 (옷/배경 색이 비슷할 때)
 *      b. 색 경계인데 모델이 "경계 아님"이라고 하면 두 구역을 합침 (그림자·조명 얼룩)
 * 5. 너무 작은 구역은 가장 길게 맞닿은 이웃 구역으로 흡수
 * 6. 구역 경계 = 검은 선. 모델의 강한 선(눈·코·입 등)은 1px 뼈대로 깎고 끊긴 끝을 이어 장식선으로 추가
 *
 * 칠하기는 구역 단위라서 장식선이 끊겨 있어도 색이 새지 않음
 */

const EDGE_CUT    = 0.6;   // 이 이상이면 같은 색이어도 구역을 자름
const EDGE_MERGE  = 0.2;   // 색 경계를 따라 모델 확률 평균이 이보다 낮으면 합침
const EDGE_DECOR  = 0.7;   // 장식선으로 그릴 최소 확률 (얇게 만든 뒤)
const EDGE_SEARCH = 2;     // 모델 선과 색 경계의 위치 오차 허용 (픽셀)
const BG_EDGE     = 0.5;   // 사람 바깥(배경) 엣지에 곱하는 값 → 배경 잔선이 줄어듦
const PERSON_PAD  = 4;     // 사람 마스크를 이만큼(px) 넓게 봄 (마스크 가장자리 오차)
const LINK_DIST   = 20;    // 장식선 끊긴 곳을 이을 최대 거리(px)
const LINK_ANGLE  = 70;    // 끝점↔끝점: 두 선의 진행 방향 허용 각도(°)
const RAY_ANGLE   = 20;    // 끝점→선: 진행 방향 기준 부채꼴 반각(°)
const LINK_BACK   = 6;     // 끝점 방향을 잴 때 선을 따라 거슬러 가는 거리(px)

export function buildColoringBook(imageData, { K = 7, minAreaRatio = 0.002, edges = null, person = null } = {}) {
  const { width: w, height: h } = imageData;
  const N = w * h;
  if (edges && person) edges = applyPersonMask(edges, person, w, h);

  const rgb = boxBlur(imageData.data, w, h, 2);
  const { centers, labels } = kmeans(rgb, N, K);

  modeFilter(labels, w, h, K);
  modeFilter(labels, w, h, K);

  // 4. 구역
  const cut = new Uint8Array(N);
  if (edges) for (let i = 0; i < N; i++) cut[i] = edges[i] > EDGE_CUT ? 1 : 0;
  const comp = components(labels, w, h, cut);
  growIntoCut(comp.map, w, h, edges);

  const uf = new UnionFind(comp.count);
  if (edges) mergeWeakBoundaries(comp.map, w, h, maxFilter(edges, w, h, EDGE_SEARCH), uf);

  // 5. 작은 구역 흡수
  const minArea = Math.max(40, Math.floor(N * minAreaRatio));
  for (let pass = 0; pass < 6; pass++) {
    if (!absorbSmallRegions(comp.map, w, h, uf, minArea)) break;
  }

  const { regionMap, count } = relabel(comp.map, uf);

  // 구역별 픽셀 목록 (채우기를 빠르게)
  const sizes = new Int32Array(count);
  for (let i = 0; i < N; i++) sizes[regionMap[i]]++;
  const regionStart = new Int32Array(count + 1);
  for (let r = 0; r < count; r++) regionStart[r + 1] = regionStart[r] + sizes[r];
  const regionPixels = new Int32Array(N);
  const cursor = regionStart.slice(0, count);
  for (let i = 0; i < N; i++) regionPixels[cursor[regionMap[i]]++] = i;

  // 6. 선: 오른쪽/아래 픽셀과 구역이 다르면 선 + 모델 장식선
  const line = new Uint8Array(N);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, r = regionMap[i];
      if ((x + 1 < w && regionMap[i + 1] !== r) || (y + 1 < h && regionMap[i + w] !== r)) line[i] = 1;
    }
  }
  if (edges) addDecorLines(line, edges, w, h, Math.max(20, Math.floor(N * 0.00015)));

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

/** 4방향 연결 요소 라벨링. cut 픽셀은 어느 구역에도 넣지 않음(-1) */
function components(labels, w, h, cut) {
  const N = w * h;
  const map = new Int32Array(N).fill(-1);
  const stack = new Int32Array(N);
  let count = 0;
  for (let i = 0; i < N; i++) {
    if (map[i] !== -1 || cut[i]) continue;
    const l = labels[i];
    let sp = 0;
    stack[sp++] = i; map[i] = count;
    while (sp) {
      const p = stack[--sp];
      const x = p % w;
      if (x > 0     && map[p-1] === -1 && !cut[p-1] && labels[p-1] === l) { map[p-1] = count; stack[sp++] = p-1; }
      if (x < w - 1 && map[p+1] === -1 && !cut[p+1] && labels[p+1] === l) { map[p+1] = count; stack[sp++] = p+1; }
      if (p >= w    && map[p-w] === -1 && !cut[p-w] && labels[p-w] === l) { map[p-w] = count; stack[sp++] = p-w; }
      if (p + w < N && map[p+w] === -1 && !cut[p+w] && labels[p+w] === l) { map[p+w] = count; stack[sp++] = p+w; }
    }
    count++;
  }
  if (count === 0) { map.fill(0); count = 1; }   // 전부 경계로 나온 극단적인 경우
  return { map, count };
}

/**
 * 경계(cut) 픽셀을 이웃 구역으로 채움. 엣지 확률이 낮은 픽셀부터 채우므로(watershed)
 * 양쪽 구역이 모델 선의 가장 진한 능선에서 만나 경계가 매끄러워짐
 */
function growIntoCut(map, w, h, edges) {
  const N = w * h, LEVELS = 64;
  const level = i => edges ? Math.min(LEVELS - 1, Math.floor(edges[i] * LEVELS)) : 0;
  const buckets = Array.from({ length: LEVELS }, () => []);
  for (let i = 0; i < N; i++) if (map[i] !== -1) buckets[0].push(i);
  for (let lv = 0; lv < LEVELS; lv++) {
    const b = buckets[lv];
    for (let k = 0; k < b.length; k++) {   // 처리 중에 같은 단계로 추가될 수 있음
      const p = b[k], r = map[p], x = p % w;
      const visit = q => { if (map[q] === -1) { map[q] = r; buckets[Math.max(lv, level(q))].push(q); } };
      if (x > 0) visit(p - 1);
      if (x < w - 1) visit(p + 1);
      if (p >= w) visit(p - w);
      if (p + w < N) visit(p + w);
    }
    b.length = 0;
  }
}

class UnionFind {
  constructor(n) { this.parent = new Int32Array(n); for (let i = 0; i < n; i++) this.parent[i] = i; }
  find(a) {
    const p = this.parent;
    while (p[a] !== a) { p[a] = p[p[a]]; a = p[a]; }
    return a;
  }
  union(a, b) { a = this.find(a); b = this.find(b); if (a !== b) this.parent[a] = b; return a !== b; }
}

/** 이웃한 두 구역(대표 번호 기준) 사이 경계 픽셀 쌍마다 fn(a, b, i) 호출 */
function forEachBoundary(map, w, h, uf, fn) {
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, a = uf.find(map[i]);
      if (x + 1 < w) { const b = uf.find(map[i + 1]); if (a !== b) fn(a, b, i); }
      if (y + 1 < h) { const b = uf.find(map[i + w]); if (a !== b) fn(a, b, i); }
    }
  }
}

/** 색 경계지만 모델이 경계로 보지 않는 곳(그림자 등) → 두 구역 합침 */
function mergeWeakBoundaries(map, w, h, edgeNear, uf) {
  const n = uf.parent.length;
  const stats = new Map();   // a*n+b → [모델 확률 합, 픽셀 수]
  forEachBoundary(map, w, h, uf, (a, b, i) => {
    const key = a < b ? a * n + b : b * n + a;
    let s = stats.get(key);
    if (!s) stats.set(key, s = [0, 0]);
    s[0] += edgeNear[i]; s[1]++;
  });
  for (const [key, [sum, cnt]] of stats) {
    if (sum / cnt < EDGE_MERGE) uf.union(Math.floor(key / n), key % n);
  }
}

/** 작은 구역을 가장 길게 맞닿은 이웃 구역에 합침. 바뀐 게 있으면 true */
function absorbSmallRegions(map, w, h, uf, minArea) {
  const n = uf.parent.length;
  const size = new Int32Array(n);
  for (let i = 0; i < map.length; i++) size[uf.find(map[i])]++;

  const contact = new Map();   // 작은 구역 → Map(이웃 → 맞닿은 길이)
  const add = (s, o) => {
    if (size[s] >= minArea) return;
    let m = contact.get(s);
    if (!m) contact.set(s, m = new Map());
    m.set(o, (m.get(o) ?? 0) + 1);
  };
  forEachBoundary(map, w, h, uf, (a, b) => { add(a, b); add(b, a); });
  if (!contact.size) return false;

  const target = [];
  for (const [s, m] of contact) {
    let best = -1, bv = 0;
    for (const [o, v] of m) if (v > bv) { bv = v; best = o; }
    target.push([s, best]);
  }
  let changed = false;
  for (const [s, t] of target) changed = uf.union(s, t) || changed;
  return changed;
}

/** 대표 번호 → 0부터 연속된 구역 번호 */
function relabel(map, uf) {
  const N = map.length;
  const id = new Int32Array(uf.parent.length).fill(-1);
  const regionMap = new Int32Array(N);
  let count = 0;
  for (let i = 0; i < N; i++) {
    const r = uf.find(map[i]);
    if (id[r] === -1) id[r] = count++;
    regionMap[i] = id[r];
  }
  return { regionMap, count };
}

/** (2r+1)x(2r+1) 최대값 필터 (가로·세로 분리) */
function maxFilter(src, w, h, r) {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let m = 0;
      for (let k = Math.max(0, x - r); k <= Math.min(w - 1, x + r); k++) m = Math.max(m, src[y * w + k]);
      tmp[y * w + x] = m;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let m = 0;
      for (let k = Math.max(0, y - r); k <= Math.min(h - 1, y + r); k++) m = Math.max(m, tmp[k * w + x]);
      out[y * w + x] = m;
    }
  }
  return out;
}

/**
 * 사람 마스크로 엣지 보정
 *  - 사람 바깥: 엣지 × BG_EDGE → 배경의 약한 선(질감)은 합쳐지고 아주 강한 선만 남음
 *  - 사람 외곽선: 엣지 = 1 → 사람 윤곽은 항상 닫힌 구역 경계가 됨
 */
function applyPersonMask(edges, person, w, h) {
  const N = w * h;
  const near = maxFilter(person, w, h, PERSON_PAD);
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) out[i] = near[i] > 0.5 ? edges[i] : edges[i] * BG_EDGE;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, inside = person[i] > 0.5;
      if ((x + 1 < w && (person[i + 1] > 0.5) !== inside) || (y + 1 < h && (person[i + w] > 0.5) !== inside)) out[i] = 1;
    }
  }
  return out;
}

/** Zhang-Suen 세선화: 굵은 선 마스크를 가운데 1px 뼈대로 깎음 (제자리 수정) */
function skeletonize(m, w, h) {
  const del = [];
  for (let changed = true; changed;) {
    changed = false;
    for (let step = 0; step < 2; step++) {
      del.length = 0;
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const i = y * w + x;
          if (!m[i]) continue;
          // P2..P9: 위부터 시계방향
          const p2 = m[i - w], p3 = m[i - w + 1], p4 = m[i + 1], p5 = m[i + w + 1],
                p6 = m[i + w], p7 = m[i + w - 1], p8 = m[i - 1], p9 = m[i - w - 1];
          const B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
          if (B < 2 || B > 6) continue;
          const A = (!p2 && p3) + (!p3 && p4) + (!p4 && p5) + (!p5 && p6) +
                    (!p6 && p7) + (!p7 && p8) + (!p8 && p9) + (!p9 && p2);
          if (A !== 1) continue;
          if (step === 0 ? (p2 && p4 && p6) || (p4 && p6 && p8) : (p2 && p4 && p8) || (p2 && p6 && p8)) continue;
          del.push(i);
        }
      }
      for (const i of del) m[i] = 0;
      if (del.length) changed = true;
    }
  }
}

/**
 * 끊긴 선 잇기 (1px 뼈대 m을 제자리 수정. 가까운 것부터, 끝점당 한 번)
 *  1) 끝점↔끝점: LINK_DIST 안이고 두 선이 서로를 향해 뻗어 있으면(LINK_ANGLE 안) 직선 연결
 *  2) 끝점→선: 남은 끝점에서 진행 방향 ±RAY_ANGLE 부채꼴로 LINK_DIST 안에 다른 선이 있으면 거기까지 (T자 이음)
 */
function linkEndpoints(m, w, h) {
  const at = (x, y) => x >= 0 && y >= 0 && x < w && y < h && m[y * w + x];
  const isEnd = (x, y) => {
    let c = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if ((dx || dy) && at(x + dx, y + dy)) c++;
    return c === 1;
  };
  // 끝점마다: 진행 방향 + 자기 선의 가까운 부분(자기 자신에 잇지 않도록)
  const ends = [];
  const traceLen = Math.max(LINK_BACK, 2 * LINK_DIST);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!m[y * w + x] || !isEnd(x, y)) continue;
      const own = new Set([y * w + x]);
      let cx = x, cy = y, bx = x, by = y;
      for (let k = 1; k <= traceLen; k++) {
        let moved = false;
        for (let dy = -1; dy <= 1 && !moved; dy++) for (let dx = -1; dx <= 1 && !moved; dx++) {
          const nx = cx + dx, ny = cy + dy;
          if ((dx || dy) && at(nx, ny) && !own.has(ny * w + nx)) { own.add(ny * w + nx); cx = nx; cy = ny; moved = true; }
        }
        if (!moved) break;
        if (k <= LINK_BACK) { bx = cx; by = cy; }
      }
      const vx = x - bx, vy = y - by, n = Math.hypot(vx, vy);
      if (n) ends.push({ x, y, dx: vx / n, dy: vy / n, own });
    }
  }

  const used = new Uint8Array(ends.length);
  const draw = (x0, y0, x1, y1) => {   // 브레젠험 직선
    const sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, ddx = Math.abs(x1 - x0), ddy = -Math.abs(y1 - y0);
    let err = ddx + ddy;
    for (;;) {
      m[y0 * w + x0] = 1;
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 >= ddy) { err += ddy; x0 += sx; }
      if (e2 <= ddx) { err += ddx; y0 += sy; }
    }
  };

  // 1) 끝점 ↔ 끝점
  const cosLim = Math.cos(LINK_ANGLE * Math.PI / 180);
  const pairs = [];
  for (let i = 0; i < ends.length; i++) {
    for (let j = i + 1; j < ends.length; j++) {
      const a = ends[i], b = ends[j];
      const vx = b.x - a.x, vy = b.y - a.y, d = Math.hypot(vx, vy);
      if (!d || d > LINK_DIST || a.own.has(b.y * w + b.x)) continue;
      if ((a.dx * vx + a.dy * vy) / d >= cosLim && -(b.dx * vx + b.dy * vy) / d >= cosLim) pairs.push([d, i, j]);
    }
  }
  pairs.sort((p, q) => p[0] - q[0]);
  for (const [, i, j] of pairs) {
    if (used[i] || used[j]) continue;
    used[i] = used[j] = 1;
    draw(ends[i].x, ends[i].y, ends[j].x, ends[j].y);
  }

  // 2) 끝점 → 선 (부채꼴 안에서 가장 가까운 선 픽셀). 1)에서 그은 선은 대상에서 제외하려고 원본을 복사해 둠
  const orig = m.slice();
  const rays = [];
  for (let k = 0; k < 9; k++) rays.push((-RAY_ANGLE + k * RAY_ANGLE / 4) * Math.PI / 180);
  ends.forEach((e, i) => {
    if (used[i]) return;
    for (let t = 2; t <= LINK_DIST; t++) {
      for (const a of rays) {
        const c = Math.cos(a), s = Math.sin(a);
        const px = Math.round(e.x + (e.dx * c - e.dy * s) * t), py = Math.round(e.y + (e.dx * s + e.dy * c) * t);
        if (px >= 0 && py >= 0 && px < w && py < h && orig[py * w + px] && !e.own.has(py * w + px)) {
          draw(e.x, e.y, px, py);
          return;
        }
      }
    }
  });
}

/** 모델의 강한 선을 1px 뼈대로 깎고 끊긴 끝을 이어 장식선으로 추가. 구역 경계 근처와 짧은 조각은 버림 */
function addDecorLines(line, edges, w, h, minLen) {
  const N = w * h;
  const nearBoundary = maxFilter(Float32Array.from(line), w, h, 2);   // 구역 경계와 겹치지 않게
  const keep = new Uint8Array(N);
  for (let i = 0; i < N; i++) keep[i] = edges[i] >= EDGE_DECOR ? 1 : 0;
  skeletonize(keep, w, h);
  linkEndpoints(keep, w, h);
  for (let i = 0; i < N; i++) if (nearBoundary[i]) keep[i] = 0;
  // 8방향 연결 조각 크기로 잡티 제거
  const seen = new Uint8Array(N), stack = new Int32Array(N), piece = [];
  for (let i = 0; i < N; i++) {
    if (!keep[i] || seen[i]) continue;
    let sp = 0; piece.length = 0;
    stack[sp++] = i; seen[i] = 1;
    while (sp) {
      const p = stack[--sp]; piece.push(p);
      const px = p % w, py = (p - px) / w;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const qx = px + dx, qy = py + dy;
        if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
        const q = qy * w + qx;
        if (keep[q] && !seen[q]) { seen[q] = 1; stack[sp++] = q; }
      }
    }
    if (piece.length >= minLen) for (const p of piece) line[p] = 1;
  }
}
