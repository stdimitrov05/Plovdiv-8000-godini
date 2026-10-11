"""Прави 1080p видео от снимки с текст, синхронизиран глас и премиум ефекти.

Употреба:
    python make_video.py plovdiv.json

Формат на конфигурацията: виж plovdiv/plovdiv.json.
"""
import argparse
import asyncio
import hashlib
import json
import math
import subprocess
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager
from pathlib import Path

import edge_tts
import imageio_ffmpeg
import numpy as np
from proglog import ProgressBarLogger
from moviepy import (
    AudioArrayClip,
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
)
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
from moviepy.video.fx import CrossFadeIn, CrossFadeOut, FadeIn, FadeOut
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

W, H = 1920, 1080
LEAD = 0.6  # пауза между началото на сцената и първата дума
BAND_H = 400  # височина на тъмната лента в долната част
CAPTION_Y = H - 175
TITLE_SECONDS = 10.0  # заглавието се показва толкова секунди от началото на гласа

DEFAULTS = {
    "output": "video.mp4",
    "fps": 30,
    "total_seconds": None,  # ако е зададено, видеото е точно толкова секунди
    "seconds_per_slide": 5,
    "transition": 1.0,
    "zoom": 0.07,
    "voice": "bg-BG-KalinaNeural",  # bg-BG-BorislavNeural е мъжки глас
    "voice_rate": "+0%",
    "music": True,
    "music_volume": 0.16,
    "font": "C:/Windows/Fonts/arial.ttf",
    "font_bold": "C:/Windows/Fonts/arialbd.ttf",
    "gold": [212, 175, 90],
    "codec": "libx264",  # "h264_nvenc" използва видео картата (NVIDIA)
    "render_chunks": 8,  # колко части се рендерират паралелно на отделни процесори
}

MEASURE = ImageDraw.Draw(Image.new("RGBA", (1, 1)))


def wrap(text, font, max_width):
    lines = []
    for paragraph in (text or "").split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if MEASURE.textlength(candidate, font=font) <= max_width:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


# ---------- Гласът и синхронизацията с текста ----------

async def _synthesize(text, cfg, mp3_path):
    communicate = edge_tts.Communicate(
        text, cfg["voice"], rate=cfg["voice_rate"], boundary="WordBoundary"
    )
    words = []
    with open(mp3_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 1e7  # edge-tts дава времена в 100 наносекунди
                end = (chunk["offset"] + chunk["duration"]) / 1e7
                words.append({"text": chunk["text"], "start": start, "end": end})
    return words


def get_voice(text, cfg, cache_dir):
    """Синтезира репликата веднъж и я кешира, за да не се прави отново при всеки рендер."""
    key = hashlib.sha1(
        json.dumps([text, cfg["voice"], cfg["voice_rate"]], ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    mp3 = cache_dir / f"{key}.mp3"
    meta = cache_dir / f"{key}.json"
    if not (mp3.exists() and meta.exists()):
        for attempt in range(1, 6):  # услугата понякога връща временна грешка 403
            try:
                words = asyncio.run(_synthesize(text, cfg, mp3))
                break
            except Exception as error:
                if attempt == 5:
                    raise
                print(f"Гласът не отговори ({type(error).__name__}), опит {attempt + 1}/5...", flush=True)
                time.sleep(5 * attempt)
        meta.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    return mp3, json.loads(meta.read_text(encoding="utf-8"))


def build_phrases(narration, words):
    """Групира думите в кратки фрази с начало и край (в секунди от началото на гласа)."""
    tokens = narration.split()
    has_letters = lambda t: any(c.isalnum() for c in t)  # noqa: E731
    timed = []  # (показван текст, начало, край); пунктуацията няма време
    if sum(has_letters(t) for t in tokens) == len(words):
        spoken = iter(words)
        for tok in tokens:
            if has_letters(tok):
                w = next(spoken)
                timed.append((tok, w["start"], w["end"]))
            else:
                timed.append((tok, None, None))
    else:  # ако думите не съвпадат, показваме думите от синтезатора
        timed = [(w["text"], w["start"], w["end"]) for w in words]

    groups, buf = [], []
    for tok, s, e in timed:
        buf.append((tok, s, e))
        count = sum(1 for x in buf if x[1] is not None)
        ends_phrase = s is not None and tok[-1] in ".!?;:,"
        if (ends_phrase and count >= 3) or count >= 8:
            groups.append(buf)
            buf = []
    if buf:
        groups.append(buf)

    phrases = []
    for g in groups:
        timed_words = [x for x in g if x[1] is not None]
        if timed_words:
            text = " ".join(x[0] for x in g).strip()
            phrases.append((text, timed_words[0][1], timed_words[-1][2]))
    return phrases


# ---------- Визия ----------

def grade(img):
    """Топъл, контрастен вид и лека винетка - за премиум усещане."""
    img = ImageEnhance.Contrast(img).enhance(1.08)
    img = ImageEnhance.Color(img).enhance(1.06)
    arr = np.asarray(img).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
    arr *= (1 - 0.35 * np.clip(r, 0, 1) ** 2)[..., None]
    arr[..., 0] *= 1.03
    arr[..., 2] *= 0.97
    return np.clip(arr, 0, 255).astype(np.uint8)


def make_background(img):
    """Вписва цялата снимка; празнините запълва с размазана версия на същата снимка."""
    cover = max(W / img.width, H / img.height)
    bg = img.resize((round(img.width * cover), round(img.height * cover)))
    left, top = (bg.width - W) // 2, (bg.height - H) // 2
    bg = bg.crop((left, top, left + W, top + H)).filter(ImageFilter.GaussianBlur(30))

    fit = min(W / img.width, H / img.height)
    fg = img.resize((round(img.width * fit), round(img.height * fit)), Image.LANCZOS)
    bg.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2))
    return bg


def band_array():
    """Тъмна лента с плавен преход към прозрачно - за по-добра четимост на текста."""
    band = np.zeros((BAND_H, W, 4), dtype=np.uint8)
    band[..., 3] = np.linspace(0, 205, BAND_H).astype(np.uint8)[:, None]
    return band


def sweep_clip(start, length):
    """Мек светлинен преход, който минава през кадъра в началото на сцената."""
    w = 420
    alpha = np.clip(1 - np.abs(np.arange(w) - w / 2) / (w / 2), 0, 1) ** 2 * 0.28
    arr = np.zeros((H, w, 4), dtype=np.uint8)
    arr[..., :3] = 255
    arr[..., 3] = (np.broadcast_to(alpha, (H, w)) * 255).astype(np.uint8)
    travel = W + w
    return (
        ImageClip(arr)
        .with_position(lambda t: (-w + travel * min(t / length, 1), 0))
        .with_start(start)
        .with_duration(length)
    )


def title_block(title, tagline, cfg):
    """Златна линия, заглавие и подзаглавие в лентата."""
    title_font = ImageFont.truetype(cfg["font_bold"], 76)
    tag_font = ImageFont.truetype(cfg["font"], 40)
    gold = tuple(cfg["gold"])
    img = Image.new("RGBA", (W, BAND_H - 30), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([W // 2 - 90, 0, W // 2 + 90, 3], fill=gold + (255,))

    y = 30
    for line in wrap(title, title_font, W - 240):
        draw.text((W // 2 + 2, y + 3), line, font=title_font, fill=(0, 0, 0, 170), anchor="mt")
        draw.text((W // 2, y), line, font=title_font, fill=(255, 255, 255, 255), anchor="mt")
        y += 92
    if title and tagline:
        y += 8
    for line in wrap(tagline, tag_font, W - 240):
        draw.text((W // 2, y), line, font=tag_font, fill=gold + (255,), anchor="mt")
        y += 56
    return np.array(img)


def caption_block(text, cfg):
    """Редът с текущата фраза от реплика, която се чува."""
    font = ImageFont.truetype(cfg["font"], 46)
    lines = wrap(text, font, W - 300)
    gap = 60
    img = Image.new("RGBA", (W, gap * len(lines) + 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = 0
    for line in lines:
        draw.text((W // 2 + 2, y + 3), line, font=font, fill=(0, 0, 0, 190), anchor="mt")
        draw.text((W // 2, y), line, font=font, fill=(255, 255, 255, 255), anchor="mt")
        y += gap
    return np.array(img)


def make_credits_overlay(lines, cfg):
    """Затъмнен кадър с центрирани редове (първият ред е заглавие)."""
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 190))
    draw = ImageDraw.Draw(overlay)

    head_font = ImageFont.truetype(cfg["font_bold"], 64)
    body_font = ImageFont.truetype(cfg["font"], 36)
    rows = []
    for i, line in enumerate(lines):
        font, step = (head_font, 90) if i == 0 else (body_font, 56)
        for part in wrap(line, font, W - 240):
            rows.append((part, font, step))
        rows.append(("", body_font, 26))

    y = (H - sum(step for _, _, step in rows)) // 2
    for part, font, step in rows:
        draw.text((W // 2, y), part, font=font, fill="white", anchor="mt")
        y += step

    return np.array(overlay)


_rng = np.random.default_rng()


def old_film(frame):
    """Стар филм: сепия, зърно, мъждукане, драскотини и прах (за една кадър)."""
    f = frame.astype(np.float32)
    gray = f @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    out = np.stack([gray * 1.08 + 10, gray * 0.88 + 4, gray * 0.66 - 4], axis=-1)
    out += _rng.standard_normal((H, W, 1), dtype=np.float32) * 10  # зърно
    out *= 1 + _rng.uniform(-0.05, 0.05)  # мъждукане
    for _ in range(int(_rng.integers(0, 3))):  # вертикални драскотини
        out[:, int(_rng.integers(0, W))] = 215
    for _ in range(int(_rng.integers(0, 8))):  # прах
        y, x = int(_rng.integers(0, H)), int(_rng.integers(0, W))
        out[y:y + 3, x:x + 3] = 235
    return np.clip(out, 0, 255).astype(np.uint8)


EFFECTS = {"old_film": old_film}


def make_photo_track(paths, duration, cfg, effect=None):
    """Снимките на една сцена се сменят с плавен преход; общо траят duration секунди."""
    t, zoom = cfg["transition"], cfg["zoom"]
    n = len(paths)
    per = (duration + (n - 1) * t) / n  # припокриването при преходите се добавя към времето
    clips = []
    for path in paths:
        frame = grade(make_background(Image.open(path).convert("RGB")))
        photo = (
            ImageClip(frame)
            .with_duration(per)
            .resized(lambda tt, per=per: 1 + zoom * tt / per)  # бавен Ken Burns ефект
            .with_position("center")
        )
        clip = CompositeVideoClip([photo], size=(W, H)).with_duration(per)
        if effect in EFFECTS:
            clip = clip.image_transform(EFFECTS[effect])
        clips.append(clip)

    if n == 1:
        return clips[0]
    clips = [c.with_effects([CrossFadeIn(t), CrossFadeOut(t)]) for c in clips]
    return concatenate_videoclips(clips, method="compose", padding=-t)


def make_slide(slide, cfg, mp3, words):
    duration = slide["duration"]
    t0 = LEAD if mp3 else 0.0  # текстът се появява с началото на гласа

    paths = slide.get("images") or [slide["image"]]
    layers = [make_photo_track(paths, duration, cfg, slide.get("effect"))]

    if slide.get("credits"):
        layers.append(
            ImageClip(make_credits_overlay(slide["credits"], cfg))
            .with_duration(duration)
            .with_effects([CrossFadeIn(0.8)])
        )
    else:
        span = duration - t0
        layers.append(
            ImageClip(band_array())
            .with_position((0, H - BAND_H))
            .with_start(t0)
            .with_duration(span)
            .with_effects([CrossFadeIn(0.8)])
        )
        layers.append(sweep_clip(0.3, 1.6))
        title_y = H - BAND_H + 30
        layers.append(
            ImageClip(title_block(slide.get("title"), slide.get("text"), cfg))
            .with_position(lambda t: (0, title_y + 50 * max(0.0, 1 - t / 0.8)))  # влиза отдолу
            .with_start(t0)
            .with_duration(min(span, TITLE_SECONDS))
            .with_effects([CrossFadeIn(0.8), CrossFadeOut(0.8)])  # след 10 секунди изчезва
        )
        if mp3:
            for text, start, end in build_phrases(slide["narration"], words):
                begin = t0 + start
                stop = min(duration, t0 + end + 0.25)
                layers.append(
                    ImageClip(caption_block(text, cfg))
                    .with_position((0, CAPTION_Y))
                    .with_start(begin)
                    .with_duration(stop - begin)
                    .with_effects([CrossFadeIn(0.25)])
                )

    clip = CompositeVideoClip(layers, size=(W, H)).with_duration(duration)
    if mp3:
        voice = AudioFileClip(str(mp3)).with_start(t0)
        clip = clip.with_audio(CompositeAudioClip([voice]).with_duration(duration))
    return clip


# ---------- Фонова музика и финал ----------

def pad_music(duration, sr=44100):
    """Тиха синтезирана подложка: акорди, които се сменят на всеки 12 секунди."""
    chords = [
        (110.0, 130.81, 164.81),  # Ам
        (87.31, 110.0, 130.81),  # F
        (130.81, 164.81, 196.0),  # C
        (98.0, 123.47, 146.83),  # G
    ]
    n = int(duration * sr)
    out = np.zeros(n, dtype=np.float32)
    seg, fade = 12.0, 1.5
    k = 0
    while k * seg < duration:
        a = k * seg
        b = min(duration, (k + 1) * seg + fade)  # съседните акорди се припокриват
        i0, i1 = int(a * sr), min(n, int(b * sr))
        tt = np.arange(i0, i1) / sr
        sig = np.zeros(i1 - i0, dtype=np.float32)
        for f in chords[k % len(chords)]:
            sig += np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt)
        env = np.clip((tt - a) / fade, 0, 1) * np.clip((b - tt) / fade, 0, 1)
        out[i0:i1] += sig * env / 3.9
        k += 1
    out *= 0.3 * (0.85 + 0.15 * np.sin(2 * np.pi * 0.15 * np.arange(n) / sr))
    return np.column_stack([out, out]).astype(np.float32)


def finish_chunk(clip, cfg, a, b, total, sr=44100):
    """Подложка и затихвания за парче [a, b) от цялото видео."""
    if cfg["music"]:
        full = pad_music(total, sr)
        music = AudioArrayClip(full[int(a * sr):int(b * sr)], fps=sr).with_volume_scaled(cfg["music_volume"])
        if clip.audio is not None:
            clip = clip.with_audio(CompositeAudioClip([clip.audio, music]).with_duration(b - a))
        else:
            clip = clip.with_audio(music.with_duration(b - a))

    # Затихването е само в началото и края на цялото видео, не на всяко парче
    video_fx, audio_fx = [], []
    if a <= 0:
        video_fx.append(FadeIn(1.0))
        audio_fx.append(AudioFadeIn(1.0))
    if b >= total - 1e-6:
        video_fx.append(FadeOut(1.5))
        audio_fx.append(AudioFadeOut(1.5))
    if video_fx:
        clip = clip.with_effects(video_fx)
    if clip.audio is not None and audio_fx:
        clip = clip.with_audio(clip.audio.with_effects(audio_fx))
    return clip.with_duration(b - a)


def build_timeline(cfg, prepared):
    T = cfg["transition"]
    clips = [make_slide(s, cfg, mp3, words) for s, mp3, words in prepared]
    clips = [c.with_effects([CrossFadeIn(T), CrossFadeOut(T)]) for c in clips]
    return concatenate_videoclips(clips, method="compose", padding=-T)


def load_plan(cfg_path):
    """Проверява гласа и слотовете; връща настройките, сцените и общата продължителност."""
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = {**DEFAULTS, **{k: v for k, v in data.items() if k != "slides"}}
    base_dir = cfg_path.parent
    cache = base_dir / "tts_cache"
    cache.mkdir(exist_ok=True)
    T = cfg["transition"]

    prepared = []
    for i, slide in enumerate(data["slides"], 1):
        slide = {
            **slide,
            "image": base_dir / slide["image"],
            "images": [base_dir / p for p in slide.get("images", [])],
            "duration": slide.get("duration", cfg["seconds_per_slide"]),
        }
        mp3, words = None, None
        if slide.get("narration"):
            mp3, words = get_voice(slide["narration"], cfg, cache)
            with AudioFileClip(str(mp3)) as a:
                length = a.duration
            if LEAD + length + 0.8 > slide["duration"]:
                raise SystemExit(
                    f"Сцена {i}: гласът трае {length:.1f} сек., а слотът е {slide['duration']} сек. "
                    "Скъсете текста или увеличете duration."
                )
        prepared.append((slide, mp3, words))

    total = sum(s["duration"] for s, _, _ in prepared) - (len(prepared) - 1) * T
    if cfg["total_seconds"] and abs(total - cfg["total_seconds"]) > 0.01:
        raise SystemExit(
            f"Общата продължителност ще е {total} сек., а трябва да е {cfg['total_seconds']} сек. "
            "Коригирайте duration на сцените."
        )
    return cfg, base_dir, prepared, total


class ChunkLogger(ProgressBarLogger):
    """Докладва броя изрендерени кадри на парчето към главния процес."""

    def __init__(self, progress, index):
        super().__init__()
        self.progress = progress
        self.index = index

    def bars_callback(self, bar, attr, value, old_value=None):
        if bar == "frame_index" and attr == "index":  # аудиото има своя лента и не се брои
            self.progress[self.index] = (value, self.bars[bar]["total"])


def render_chunk(job):
    """Рендерира едно парче от видеото в отделен процес (паралелно с останалите)."""
    cfg_path, a, b, index, tmp_dir, progress = job
    cfg, _, prepared, total = load_plan(Path(cfg_path))
    final = build_timeline(cfg, prepared)
    part = None
    try:
        part = finish_chunk(final.subclipped(a, b), cfg, a, b, total)
        out = Path(tmp_dir) / f"part_{index:02d}.mp4"
        part.write_videofile(
            str(out),
            fps=cfg["fps"],
            codec=cfg["codec"],
            audio_codec="aac",
            bitrate="8000k",
            preset="medium",
            logger=ChunkLogger(progress, index),
        )
    finally:
        if part is not None:
            part.close()
        final.close()
    return str(out)


def fmt_time(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def scene_spans(prepared, transition):
    """Къде започва и свършва всяка сцена във видеото и как се казва."""
    spans, start = [], 0.0
    for slide, _, _ in prepared:
        end = start + slide["duration"]
        spans.append((start, end, slide.get("title") or "Авторски права"))
        start = end - transition  # сцените се припокриват при прехода
    return spans


def part_titles(spans, a, b):
    names = [name for s0, s1, name in spans if s0 < b and s1 > a]
    return " / ".join(dict.fromkeys(names))


def draw_progress(fraction, parts_done, parts_total, end=""):
    """Една ред в конзолата: лента, процент и готови части."""
    width = 40
    filled = min(width, int(fraction * width))
    line = f"\r[{'#' * filled}{'-' * (width - filled)}] {min(fraction, 1) * 100:5.1f}%  части: {parts_done}/{parts_total}"
    print(line, end=end, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Видео от снимки с текст и глас")
    parser.add_argument("config", help="JSON файл с настройки и слайдове")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg, base_dir, prepared, total = load_plan(cfg_path)
    output = base_dir / cfg["output"]
    print(f"Сцени: {len(prepared)}, продължителност: {total:.2f} сек., части: {cfg['render_chunks']}")

    # Границите са по кадър, така че частите се слепват без пропуск или дублиране
    n, fps = cfg["render_chunks"], cfg["fps"]
    bounds = [round(total * i / n * fps) / fps for i in range(n)] + [total]
    total_frames = max(1, round(total * fps))
    spans = scene_spans(prepared, cfg["transition"])
    label = [
        f"{fmt_time(bounds[i])}-{fmt_time(bounds[i + 1])}  {part_titles(spans, bounds[i], bounds[i + 1])}"
        for i in range(n)
    ]
    for i in range(n):
        print(f"Част {i + 1}: {label[i]}")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, Manager() as manager:
        progress = manager.dict()
        jobs = [(str(cfg_path), bounds[i], bounds[i + 1], i, tmp, progress) for i in range(n)]
        parts = [None] * n
        with ProcessPoolExecutor(max_workers=n) as pool:
            futures = {pool.submit(render_chunk, job): job[3] for job in jobs}
            pending = set(futures)
            while pending:
                finished, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in finished:
                    i = futures[future]
                    parts[i] = future.result()  # при грешка тук се вижда трасировката
                    print(f"\nГотова част {i + 1}: {label[i]}")
                frames_done = sum(v for v, _ in list(progress.values()))
                draw_progress(frames_done / total_frames, n - len(pending), n)
        draw_progress(1.0, n, n, end="\n")

        list_file = Path(tmp) / "parts.txt"
        list_file.write_text("".join(f"file '{Path(p).as_posix()}'\n" for p in parts), encoding="utf-8")
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-c", "copy", "-movflags", "+faststart", str(output),
            ],
            check=True,
        )

    print(f"Готово: {output}")


if __name__ == "__main__":
    main()
