# 캠핑 소리 입히기 (`camping_audio.py`)

영상의 **도로 잡음을 줄이고**, 장작 타닥임 · 새소리 · 풀벌레 · 부엉이 · 바람 소리를
장면에 맞게 합성해서 입힙니다. 효과음은 코드로 직접 만들어서 저작권 걱정이 없습니다.

미리듣기: `samples/캠핑소리_미리듣기_낮-저녁-밤.m4a` (0~20초 낮 → 20~40초 저녁 → 40초~ 밤)

## 준비
- Python 3, `pip install numpy`
- ffmpeg (Windows: `winget install ffmpeg`, Mac: `brew install ffmpeg`)

## 사용법
```bash
# 기본: 영상 전체를 밤 캠핑 분위기로 (장작 + 풀벌레 + 부엉이)
python camping_audio.py "C:\Users\나\Desktop\할로윈\완성.mp4"

# 장면별로 지정 (초 단위, day / evening / night)
python camping_audio.py 완성.mp4 --scenes "0-40:day,40-90:evening,90-end:night"

# 도로 소음이 심하면 강하게, 캠핑 소리를 더 크게
python camping_audio.py 완성.mp4 --denoise strong --ambience 0.5

# 장작 소리 2배, 새소리 끄기
python camping_audio.py 완성.mp4 --fire 2 --birds 0
```
결과는 원본 옆에 `완성_캠핑.mp4`로 저장됩니다 (영상은 재인코딩 없이 그대로).

| 옵션 | 설명 |
|---|---|
| `--denoise light/normal/strong` | 도로 잡음 제거 강도 (기본 normal) |
| `--ambience 0~1` | 캠핑 소리 전체 볼륨 (기본 0.35) |
| `--fire/--birds/--crickets/--owl/--wind` | 레이어별 배율 (0이면 끔) |
| `--no-duck` | 말소리가 나올 때 배경음을 자동으로 줄이는 기능 끄기 |
| `--ambience-only 파일.wav` | 캠핑 소리만 따로 저장 (편집 프로그램에서 직접 섞을 때) |
