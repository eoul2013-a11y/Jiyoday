#!/usr/bin/env python3
"""영상의 도로 잡음을 줄이고 캠핑 분위기 소리(장작, 새, 풀벌레, 부엉이, 바람)를 입힌다.

필요한 것: Python 3 + numpy, ffmpeg (PATH에 있어야 함)

사용 예:
  python camping_audio.py 할로윈.mp4
  python camping_audio.py 할로윈.mp4 --scenes "0-40:day,40-90:evening,90-end:night"
  python camping_audio.py 할로윈.mp4 --denoise strong --ambience 0.5
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

SR = 48000

# 장면별 각 레이어 볼륨 (0~1)
SCENES = {
    "day":     {"fire": 0.55, "birds": 1.0, "crickets": 0.0, "owl": 0.0, "wind": 0.6},
    "evening": {"fire": 0.85, "birds": 0.45, "crickets": 0.5, "owl": 0.0, "wind": 0.45},
    "night":   {"fire": 1.0, "birds": 0.0, "crickets": 1.0, "owl": 1.0, "wind": 0.35},
}

DENOISE = {
    # highpass(Hz), afftdn 감소량(dB)
    "light":  (80, 12),
    "normal": (120, 20),
    "strong": (160, 30),
}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"명령 실패: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


def duration_of(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "json", path])
    return float(json.loads(out)["format"]["duration"])


def has_audio(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "a",
               "-show_entries", "stream=index", "-of", "json", path])
    return bool(json.loads(out).get("streams"))


# ---------- 소리 합성 ----------

def shaped_noise(n, rng, lo, hi, tilt=0.0):
    """lo~hi Hz 대역의 노이즈. tilt<0 이면 저음 쪽으로 기운다(-1 ≈ 브라운).
    긴 신호는 sqrt-hann 창으로 겹쳐 이어 붙여 메모리를 아낀다."""
    block = 1 << 18
    hop = block // 2
    win = np.sqrt(np.hanning(block))
    freqs = np.fft.rfftfreq(block, 1 / SR)
    mask = ((freqs >= lo) & (freqs <= hi)).astype(float)
    with np.errstate(divide="ignore"):
        mask *= np.where(freqs > 0, (np.maximum(freqs, 1) / 1000.0) ** tilt, 0)
    out = np.zeros(n + 2 * block)
    for start in range(0, n + hop, hop):
        spec = np.fft.rfft(rng.standard_normal(block)) * mask
        out[start:start + block] += np.fft.irfft(spec, block) * win
    out = out[:n]
    return out / (np.std(out) + 1e-9)


def slow_lfo(n, rng, rate, depth):
    """느리게 출렁이는 볼륨 곡선 (1-depth ~ 1)."""
    pts = max(4, int(n / SR * rate) + 4)
    knots = rng.random(pts)
    x = np.linspace(0, pts - 1, n)
    curve = np.interp(x, np.arange(pts), knots)
    return 1 - depth + depth * curve


def add_event(buf, pos, sig, pan):
    """스테레오 버퍼에 이벤트를 패닝해서 더한다. pan: -1(왼)~1(오)."""
    if pos >= buf.shape[1]:
        return
    end = min(buf.shape[1], pos + len(sig))
    sig = sig[:end - pos]
    gl = np.sqrt((1 - pan) / 2)
    gr = np.sqrt((1 + pan) / 2)
    buf[0, pos:end] += sig * gl
    buf[1, pos:end] += sig * gr


def gen_fire(n, rng):
    buf = np.zeros((2, n))
    # 웅웅거리는 불꽃 바닥 소리 + 쉬익하는 소리
    bed = shaped_noise(n, rng, 40, 600, tilt=-1.0) * slow_lfo(n, rng, 0.8, 0.5) * 0.25
    hiss = shaped_noise(n, rng, 1500, 12000, tilt=-1.5) * slow_lfo(n, rng, 1.5, 0.8) * 0.008
    buf += bed + hiss
    # 타닥타닥: 작은 튐 + 가끔 큰 '탁'
    t = 0.0
    dur = n / SR
    while t < dur:
        t += rng.exponential(1 / 7.0)
        big = rng.random() < 0.06
        length = int(SR * (rng.uniform(0.015, 0.04) if big else rng.uniform(0.002, 0.008)))
        burst = np.diff(rng.standard_normal(length + 1))  # 고역 강조
        burst *= np.exp(-np.linspace(0, 6, length))
        amp = rng.uniform(0.5, 1.0) if big else rng.uniform(0.08, 0.35)
        add_event(buf, int(t * SR), burst * amp, rng.uniform(-0.3, 0.3))
    return buf


def chirp(f0, f1, length, rng, harm=0.25):
    n = int(SR * length)
    tt = np.arange(n) / SR
    f = np.linspace(f0, f1, n) * (1 + 0.03 * np.sin(2 * np.pi * rng.uniform(20, 60) * tt))
    phase = 2 * np.pi * np.cumsum(f) / SR
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 2
    return (np.sin(phase) + harm * np.sin(2 * phase)) * env


def gen_birds(n, rng):
    buf = np.zeros((2, n))
    dur = n / SR
    # 서로 다른 '종' 몇 마리: 음높이, 노래 길이, 위치가 각자 다르다
    for _ in range(4):
        base = rng.uniform(2500, 5000)
        sweep = rng.uniform(-1800, 1800)
        clen = rng.uniform(0.04, 0.14)
        pan = rng.uniform(-0.9, 0.9)
        dist = rng.uniform(0.25, 1.0)  # 멀리 있는 새는 작게
        t = rng.uniform(0, 5)
        while t < dur:
            notes = rng.integers(2, 8)
            nt = t
            for _ in range(notes):
                f0 = base * rng.uniform(0.92, 1.08)
                sig = chirp(f0, max(800, f0 + sweep * rng.uniform(0.6, 1.2)),
                            clen * rng.uniform(0.8, 1.2), rng)
                add_event(buf, int(nt * SR), sig * 0.22 * dist, pan)
                nt += clen + rng.uniform(0.04, 0.18)
            t = nt + rng.uniform(2.5, 9)
    return buf


def gen_crickets(n, rng):
    buf = np.zeros((2, n))
    dur = n / SR
    for _ in range(6):
        fc = rng.uniform(4000, 5200)
        rate = rng.uniform(1.2, 3.0)  # 초당 울음 횟수
        pulses = rng.integers(2, 5)
        pan = rng.uniform(-1, 1)
        amp = rng.uniform(0.02, 0.06)
        plen = int(SR * 0.014)
        tt = np.arange(plen) / SR
        pulse = np.sin(2 * np.pi * fc * tt) * np.sin(np.pi * np.linspace(0, 1, plen)) ** 2
        t = rng.uniform(0, 1)
        while t < dur:
            for k in range(pulses):
                add_event(buf, int((t + k * 0.024) * SR), pulse * amp, pan)
            t += 1 / rate * rng.uniform(0.9, 1.1)
            if rng.random() < 0.02:  # 가끔 쉬었다 운다
                t += rng.uniform(2, 6)
    return buf


def gen_owl(n, rng):
    """할로윈 밤 분위기용 부엉이 '부-부우'."""
    buf = np.zeros((2, n))
    dur = n / SR
    t = rng.uniform(6, 15)
    while t < dur:
        pan = rng.uniform(-0.8, 0.8)
        f = rng.uniform(330, 420)
        nt = t
        for length in (0.25, 0.55):
            m = int(SR * length)
            freq = np.linspace(f * 1.04, f * 0.96, m)
            phase = 2 * np.pi * np.cumsum(freq) / SR
            env = np.sin(np.pi * np.linspace(0, 1, m)) ** 1.5
            sig = (np.sin(phase) + 0.15 * np.sin(2 * phase)) * env
            add_event(buf, int(nt * SR), sig * 0.12, pan)
            nt += length + 0.18
        t += rng.uniform(18, 40)
    return buf


def gen_wind(n, rng):
    w = shaped_noise(n, rng, 150, 1200, tilt=-0.5) * slow_lfo(n, rng, 0.15, 0.8) * 0.12
    w2 = shaped_noise(n, rng, 150, 1200, tilt=-0.5) * slow_lfo(n, rng, 0.15, 0.8) * 0.12
    return np.stack([w, w2])


GENERATORS = {"fire": gen_fire, "birds": gen_birds, "crickets": gen_crickets,
              "owl": gen_owl, "wind": gen_wind}


def parse_scenes(spec, dur):
    items = []
    for part in spec.split(","):
        rng_s, name = part.strip().split(":")
        a, b = rng_s.split("-")
        a = float(a)
        b = dur if b.strip() in ("end", "") else float(b)
        if name not in SCENES:
            sys.exit(f"알 수 없는 장면 '{name}'. 사용 가능: {', '.join(SCENES)}")
        items.append((a, min(b, dur), name))
    return items


def layer_envelopes(scenes, n, fade=3.0):
    """장면 타임라인 → 레이어별 볼륨 곡선 (장면 사이 fade초 크로스페이드)."""
    t = np.arange(n) / SR
    env = {k: np.zeros(n) for k in GENERATORS}
    weight = np.zeros(n)
    for a, b, name in scenes:
        w = np.clip(np.minimum(t - a + fade / 2, b + fade / 2 - t) / fade, 0, 1)
        weight += w
        for k in GENERATORS:
            env[k] += w * SCENES[name][k]
    weight[weight == 0] = 1
    return {k: v / weight for k, v in env.items()}


def build_ambience(dur, scenes, levels, seed, path):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    envs = layer_envelopes(scenes, n)
    mix = np.zeros((2, n))
    for name, gen in GENERATORS.items():
        g = levels.get(name, 1.0)
        if g <= 0 or envs[name].max() <= 0:
            continue
        print(f"  - {name} 생성 중...")
        mix += gen(n, rng) * envs[name] * g
    # 시작/끝 페이드
    f = min(n // 2, int(SR * 2))
    ramp = np.linspace(0, 1, f)
    mix[:, :f] *= ramp
    mix[:, n - f:] *= ramp[::-1]
    mix /= max(1e-9, np.abs(mix).max()) / 0.9
    pcm = (mix.T * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---------- 메인 ----------

def main():
    p = argparse.ArgumentParser(description="도로 잡음 제거 + 캠핑 소리 입히기")
    p.add_argument("input", help="원본 영상")
    p.add_argument("-o", "--output", help="결과 파일 (기본: 원본이름_캠핑.mp4)")
    p.add_argument("--scenes", default="0-end:night",
                   help='장면 타임라인. 예: "0-40:day,40-90:evening,90-end:night" '
                        "(day/evening/night)")
    p.add_argument("--denoise", choices=list(DENOISE), default="normal",
                   help="도로 잡음 제거 강도")
    p.add_argument("--ambience", type=float, default=0.35,
                   help="캠핑 소리 전체 볼륨 (0~1, 기본 0.35)")
    p.add_argument("--no-duck", action="store_true",
                   help="말소리가 날 때 배경음을 자동으로 줄이는 기능 끄기")
    for k in GENERATORS:
        p.add_argument(f"--{k}", type=float, default=1.0, help=f"{k} 레이어 배율")
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--ambience-only", metavar="WAV",
                   help="영상 처리 없이 캠핑 소리만 WAV로 저장 (미리듣기용)")
    a = p.parse_args()

    dur = duration_of(a.input)
    scenes = parse_scenes(a.scenes, dur)
    levels = {k: getattr(a, k) for k in GENERATORS}

    with tempfile.TemporaryDirectory() as tmp:
        amb = a.ambience_only or os.path.join(tmp, "ambience.wav")
        print(f"캠핑 소리 합성 ({dur:.1f}초)")
        build_ambience(dur, scenes, levels, a.seed, amb)
        if a.ambience_only:
            print(f"완료: {amb}")
            return

        out = a.output or os.path.splitext(a.input)[0] + "_캠핑.mp4"
        hp, nr = DENOISE[a.denoise]
        amb_chain = f"[1:a]volume={a.ambience}[amb]"
        if has_audio(a.input):
            clean = (f"[0:a]aformat=sample_rates={SR}:channel_layouts=stereo,"
                     f"highpass=f={hp},highpass=f={hp},"
                     f"afftdn=nr={nr}:nf=-30:tn=1,"
                     f"lowpass=f=14000")
            if a.no_duck:
                graph = (f"{clean}[v];{amb_chain};"
                         f"[v][amb]amix=inputs=2:normalize=0[m]")
            else:
                graph = (f"{clean},asplit[v][sc];{amb_chain};"
                         f"[amb][sc]sidechaincompress=threshold=0.03:ratio=4:"
                         f"attack=20:release=400[ducked];"
                         f"[v][ducked]amix=inputs=2:normalize=0[m]")
        else:
            graph = f"{amb_chain};[amb]anull[m]"
        graph += ";[m]alimiter=limit=0.95[out]"

        print(f"도로 잡음 제거({a.denoise}) + 믹싱 → {out}")
        run(["ffmpeg", "-y", "-hide_banner", "-i", a.input, "-i", amb,
             "-filter_complex", graph, "-map", "0:v?", "-map", "[out]",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", out])
        print(f"완료: {out}")


if __name__ == "__main__":
    main()
