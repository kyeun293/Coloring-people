/**
 * coloringMode.js — 색칠 모드 상태 관리 + 화면 그리기
 *
 *  idle ──(✌️ 1초 유지)──▶ countdown(3초) ──▶ 캡쳐 + 색칠판 생성 ──▶ coloring
 *   ▲                                                                  │
 *   └──────────────────────────(✊ 1초 유지)───────────────────────────┘
 *
 * coloring 중: 팔레트 위에서 집기 → 색을 든 상태 → 색칠판 위에서 놓기 → 그 구역 채움
 * 마우스로도 테스트 가능: 팔레트 클릭 → 색칠판 클릭
 */

import { buildColoringBook } from './coloringBook.js';
import { detectEdges, preloadEdgeModel } from './edgeModel.js';
import { detectPerson, preloadPersonMask } from './personMask.js';
import { isVSign, isFist, PinchDetector, HoldTrigger, videoRect, toScreen } from './gestures.js';

const COUNTDOWN_MS   = 3000;
const CAPTURE_ASPECT = 4 / 3;
const PROC_W         = 480;          // 이미지 처리 해상도 (가로)
const PALETTE_Y      = 64;           // 팔레트 중심 높이
const SWATCH_R       = 30;
const SWATCH_GAP     = 84;
const ERASER         = [255, 255, 255];

const HAND_LINKS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],
];

export class ColoringMode {
  constructor(canvas, ctx, video) {
    this.canvas = canvas;
    this.ctx    = ctx;
    this.video  = video;
    this.enabled = false;
    this.startHold = new HoldTrigger(1000);
    this.endHold   = new HoldTrigger(1000);
    this.pinch     = new PinchDetector();
    this._bindMouse();
    preloadEdgeModel();
    preloadPersonMask();
    this.reset();
  }

  reset() {
    this.phase      = 'idle';
    this.board      = null;
    this.held       = null;     // 들고 있는 색 [r,g,b]
    this.mouseColor = null;     // 마우스 테스트용 선택 색
    this.buildId    = (this.buildId ?? 0) + 1;  // 생성 도중 reset되면 결과 버림
    this.cursor     = null;     // 손가락 커서 (화면 좌표, 부드럽게)
    this.pinchPos   = null;     // 집고 있는 동안의 마지막 위치 (놓을 때 이 위치에 칠함)
    this.countStart = 0;
    this.ignoreUntil = performance.now() + 800;  // 모드 전환 직후 오발동 방지
    this.startHold.reset(); this.endHold.reset(); this.pinch.reset();
  }

  /** 매 프레임 호출 */
  update(landmarks, now) {
    const { canvas, video } = this;
    const W = canvas.width, H = canvas.height;
    const vr     = videoRect(video, W, H);
    const aspect = video.videoWidth / video.videoHeight;
    const hand   = landmarks?.[0] ?? null;
    const gesturesOn = now > this.ignoreUntil;

    let progress = 0;   // 모션 유지 진행도 (0~1)

    if (this.phase === 'idle') {
      const t = this.startHold.update(gesturesOn && hand && isVSign(hand, aspect), now);
      progress = t.progress;
      if (t.fired) { this.phase = 'countdown'; this.countStart = now; }

    } else if (this.phase === 'countdown') {
      if (now - this.countStart >= COUNTDOWN_MS) this._captureAndBuild(W, H, vr);

    } else if (this.phase === 'coloring') {
      this._updateColoring(hand, aspect, vr, W, H);
      const pinching = this.pinch.on;
      const t = this.endHold.update(gesturesOn && hand && !pinching && !this.held && isFist(hand, aspect), now);
      progress = t.progress;
      if (t.fired) this.reset();
    }

    this._render(W, H, vr, landmarks, now, progress);
  }

  // ── 색칠: 집기 / 놓기 ──────────────────────────────
  _updateColoring(hand, aspect, vr, W, H) {
    if (!hand) {
      this.cursor = null; this.pinchPos = null;
      if (this.pinch.on) this.held = null;   // 집은 채로 손이 사라지면 색 내려놓음
      this.pinch.reset();
      return;
    }
    const a = toScreen(hand[4], vr), b = toScreen(hand[8], vr);
    const p = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    this.cursor = this.cursor
      ? { x: this.cursor.x * 0.5 + p.x * 0.5, y: this.cursor.y * 0.5 + p.y * 0.5 }
      : p;

    const ev = this.pinch.update(hand, aspect);
    if (ev.on) this.pinchPos = { ...this.cursor };

    if (ev.started) {
      const color = this._swatchAt(this.cursor, W);
      if (color) this.held = color;
    }
    if (ev.ended) {
      if (this.held && this.pinchPos) this._fillAtScreen(this.pinchPos, W, H);
      this.held = null;
      this.pinchPos = null;
    }
  }

  _fillAtScreen(pt, W, H) {
    const r = this._boardRect(W, H);
    this.board.fillAt((pt.x - r.x) / r.w, (pt.y - r.y) / r.h, this.held);
  }

  // ── 캡쳐 → 색칠판 ─────────────────────────────────
  _captureAndBuild(W, H, vr) {
    const { video } = this;
    const cr = this._captureRect(W, H);
    const vw = video.videoWidth, vh = video.videoHeight;

    // 화면상의 캡쳐 영역 → 원본 영상 좌표 (영상은 좌우 반전 표시)
    const sw = cr.w / vr.w * vw, sh = cr.h / vr.h * vh;
    const sx = vw * (1 - (cr.x + cr.w - vr.x) / vr.w);
    const sy = (cr.y - vr.y) / vr.h * vh;

    const pw = PROC_W, ph = Math.round(PROC_W / CAPTURE_ASPECT);
    const off = document.createElement('canvas');
    off.width = pw; off.height = ph;
    const c = off.getContext('2d', { willReadFrequently: true });
    c.translate(pw, 0); c.scale(-1, 1);   // 화면에 보이는 그대로
    c.drawImage(video, sx, sy, sw, sh, 0, 0, pw, ph);
    const img = c.getImageData(0, 0, pw, ph);

    this.phase = 'building';
    const id = this.buildId;
    // "만드는 중" 화면이 한 번 그려진 뒤 처리 (처리 중에는 화면이 멈춤)
    setTimeout(async () => {
      if (id !== this.buildId) return;
      const t0 = performance.now();
      // 학습된 엣지 모델 + 사람 분할 (각각 없으면 null → 그 단계만 건너뜀)
      const [edges, person] = await Promise.all([detectEdges(img), detectPerson(off)]);
      if (id !== this.buildId) return;
      const t1 = performance.now();
      this.board = buildColoringBook(img, { edges, person });
      console.log(`[coloring] 엣지 모델 ${edges ? '사용' : '미사용'}, 사람 마스크 ${person ? '사용' : '미사용'} ` +
                  `(${Math.round(t1 - t0)}ms), ` +
                  `색칠판 생성 ${Math.round(performance.now() - t1)}ms, 팔레트 ${this.board.palette.length}색`);
      this.phase = 'coloring';
      this.ignoreUntil = performance.now() + 800;
    }, 30);
  }

  // ── 레이아웃 ───────────────────────────────────────
  _captureRect(W, H) {
    let h = H * 0.72, w = h * CAPTURE_ASPECT;
    if (w > W * 0.85) { w = W * 0.85; h = w / CAPTURE_ASPECT; }
    return { x: (W - w) / 2, y: (H - h) / 2, w, h };
  }

  _boardRect(W, H) {
    const top = PALETTE_Y + SWATCH_R + 24, bottom = H - 90;
    let h = bottom - top, w = h * CAPTURE_ASPECT;
    if (w > W - 80) { w = W - 80; h = w / CAPTURE_ASPECT; }
    return { x: (W - w) / 2, y: top + (bottom - top - h) / 2, w, h };
  }

  _swatches(W) {
    const colors = [...this.board.palette, ERASER];
    const x0 = W / 2 - (colors.length - 1) * SWATCH_GAP / 2;
    return colors.map((color, i) => ({ color, x: x0 + i * SWATCH_GAP, y: PALETTE_Y, eraser: i === colors.length - 1 }));
  }

  _swatchAt(pt, W) {
    if (!pt || !this.board) return null;
    const s = this._swatches(W).find(s => Math.hypot(pt.x - s.x, pt.y - s.y) < SWATCH_R + 14);
    return s ? s.color : null;
  }

  // ── 마우스 테스트 ──────────────────────────────────
  _bindMouse() {
    window.addEventListener('pointerdown', e => {
      if (!this.enabled || this.phase !== 'coloring' || e.target.closest('button')) return;
      const W = this.canvas.width, H = this.canvas.height;
      const pt = { x: e.clientX, y: e.clientY };
      const color = this._swatchAt(pt, W);
      if (color) { this.mouseColor = color; return; }
      if (this.mouseColor) {
        const prev = this.held; this.held = this.mouseColor;
        this._fillAtScreen(pt, W, H);
        this.held = prev;
      }
    });
  }

  // ── 그리기 ─────────────────────────────────────────
  _render(W, H, vr, landmarks, now, progress) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, W, H);

    if (this.phase === 'idle' || this.phase === 'countdown') {
      const cr = this._captureRect(W, H);
      if (this.phase === 'countdown') {
        // 캡쳐 영역 바깥 어둡게
        ctx.save();
        ctx.fillStyle = 'rgba(0,0,0,0.55)';
        ctx.beginPath(); ctx.rect(0, 0, W, H); ctx.rect(cr.x, cr.y, cr.w, cr.h);
        ctx.fill('evenodd');
        ctx.restore();
      }
      this._frame(cr);
      this._hands(landmarks, vr, 'rgba(255,255,255,0.6)');

      if (this.phase === 'idle') {
        this._label('✌️ 브이를 1초 유지하면 3초 뒤 촬영', W / 2, cr.y - 22, '#fff');
        const hand = landmarks?.[0];
        if (hand && progress > 0) this._ring(toScreen(hand[9], vr), progress, '#a78bfa');
      } else {
        const left = Math.ceil((COUNTDOWN_MS - (now - this.countStart)) / 1000);
        ctx.save();
        ctx.font = '800 160px Syne, sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillStyle = 'rgba(255,255,255,0.9)';
        ctx.shadowColor = 'rgba(139,92,246,0.9)'; ctx.shadowBlur = 40;
        ctx.fillText(String(Math.max(1, left)), W / 2, H / 2);
        ctx.restore();
      }
      return;
    }

    // building / coloring: 캠을 가림
    ctx.fillStyle = '#efece6';
    ctx.fillRect(0, 0, W, H);

    if (this.phase === 'building') {
      this._label('색칠판 만드는 중...', W / 2, H / 2, '#333');
      return;
    }

    // 색칠판
    const br = this._boardRect(W, H);
    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.18)'; ctx.shadowBlur = 24; ctx.shadowOffsetY = 6;
    ctx.fillStyle = '#fff'; ctx.fillRect(br.x, br.y, br.w, br.h);
    ctx.restore();
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(this.board.canvas, br.x, br.y, br.w, br.h);

    // 팔레트
    const hover = this.cursor && !this.held ? this._swatchAt(this.cursor, W) : null;
    for (const s of this._swatches(W)) {
      const active = s.color === this.held || s.color === hover || s.color === this.mouseColor;
      ctx.beginPath();
      ctx.arc(s.x, s.y, SWATCH_R * (active ? 1.15 : 1), 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${s.color})`;
      ctx.fill();
      ctx.lineWidth = active ? 4 : 2;
      ctx.strokeStyle = active ? '#8b5cf6' : 'rgba(0,0,0,0.25)';
      ctx.stroke();
      if (s.eraser) {
        ctx.save();
        ctx.font = '700 11px "JetBrains Mono", monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillStyle = '#666'; ctx.fillText('지우개', s.x, s.y);
        ctx.restore();
      }
    }

    // 손 + 커서
    this._hands(landmarks, vr, 'rgba(60,60,80,0.25)');
    if (this.cursor) {
      ctx.save();
      if (this.held) {
        ctx.beginPath(); ctx.arc(this.cursor.x, this.cursor.y, 18, 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${this.held})`; ctx.fill();
        ctx.lineWidth = 3; ctx.strokeStyle = '#222'; ctx.stroke();
      } else {
        ctx.beginPath(); ctx.arc(this.cursor.x, this.cursor.y, this.pinch.on ? 8 : 14, 0, Math.PI * 2);
        ctx.lineWidth = 3; ctx.strokeStyle = '#8b5cf6'; ctx.stroke();
      }
      ctx.restore();
    }

    const hand = landmarks?.[0];
    if (hand && progress > 0) this._ring(toScreen(hand[9], vr), progress, '#ef4444');

    this._label('엄지·검지로 색 집기 → 칠할 곳에서 놓기   |   ✊ 1초 유지: 종료', W / 2, H - 70, '#555', 13);
  }

  _frame(r) {
    const ctx = this.ctx;
    ctx.save();
    ctx.setLineDash([12, 8]);
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.strokeRect(r.x, r.y, r.w, r.h);
    ctx.restore();
  }

  _hands(landmarks, vr, color) {
    if (!landmarks?.length) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    for (const hand of landmarks) {
      const pts = hand.map(lm => toScreen(lm, vr));
      for (const [a, b] of HAND_LINKS) {
        ctx.beginPath(); ctx.moveTo(pts[a].x, pts[a].y); ctx.lineTo(pts[b].x, pts[b].y); ctx.stroke();
      }
    }
    ctx.restore();
  }

  _ring(c, progress, color) {
    const ctx = this.ctx;
    ctx.save();
    ctx.lineWidth = 6; ctx.lineCap = 'round';
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.beginPath(); ctx.arc(c.x, c.y, 40, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = color;
    ctx.beginPath(); ctx.arc(c.x, c.y, 40, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2); ctx.stroke();
    ctx.restore();
  }

  _label(text, x, y, color, size = 16) {
    const ctx = this.ctx;
    ctx.save();
    ctx.font = `700 ${size}px "JetBrains Mono", monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = color;
    if (color === '#fff') { ctx.shadowColor = 'rgba(0,0,0,0.8)'; ctx.shadowBlur = 6; }
    ctx.fillText(text, x, y);
    ctx.restore();
  }
}
