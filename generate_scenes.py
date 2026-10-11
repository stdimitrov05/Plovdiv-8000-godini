"""Генерира илюстрирани кадри (1920x1080) за видеото "Пловдив" с Pillow.

Това са заместители на реални снимки. Заменете файловете в scenes/
или пътищата в plovdiv.json с ваши снимки и пуснете make_video.py.
"""
import math
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

W, H = 1920, 1080
OUT = Path(__file__).parent / "plovdiv" / "scenes"


def gradient(top, bottom):
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    for y in range(H):
        t = y / (H - 1)
        color = tuple(round(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        draw.line([(0, y), (W, y)], fill=color)
    return img


def glow(img, center, radius, color, blur=120):
    layer = Image.new("RGB", (W, H), (0, 0, 0))
    x, y = center
    ImageDraw.Draw(layer).ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    return ImageChops.add(img, layer.filter(ImageFilter.GaussianBlur(blur)))


def hill_range(img, base, amp, color, phase):
    """Хълмове от синусоиди - за всеки слой се използва различна фаза."""
    draw = ImageDraw.Draw(img)
    pts = [(0, H)]
    for x in range(0, W + 10, 10):
        y = base - amp * (0.6 * math.sin(x / 330 + phase) + 0.4 * math.sin(x / 150 + phase * 2))
        pts.append((x, y))
    pts.append((W, H))
    draw.polygon(pts, fill=color)


def scene_intro():
    img = gradient((255, 170, 90), (60, 35, 90))
    img = glow(img, (1500, 560), 180, (255, 210, 130), blur=90)
    draw = ImageDraw.Draw(img)
    draw.ellipse([1380, 440, 1620, 680], fill=(255, 236, 180))
    hill_range(img, 640, 90, (120, 70, 110), 0.5)
    hill_range(img, 740, 80, (80, 48, 88), 1.7)
    hill_range(img, 860, 60, (40, 26, 58), 3.1)
    return img


def scene_antiquity():
    img = gradient((250, 215, 150), (200, 120, 80))
    draw = ImageDraw.Draw(img)
    # Амфитеатър: концентрични дъги от камъни
    cx, cy = W // 2, 800
    for i, rx in enumerate(range(700, 260, -90)):
        ry = rx * 0.35
        draw.ellipse([cx - rx, cy - ry - 180 + i * 6, cx + rx, cy + ry - 180 + i * 6],
                     outline=(230, 205, 170), width=18)
    draw.polygon([(cx - 220, 640), (cx + 220, 640), (cx + 300, 800), (cx - 300, 800)],
                 fill=(215, 190, 160))
    # Колони по краищата
    for x in list(range(120, 520, 110)) + list(range(1400, 1800, 110)):
        draw.rectangle([x, 380, x + 42, 820], fill=(240, 228, 205))
        draw.rectangle([x - 10, 370, x + 52, 395], fill=(225, 210, 185))
    draw.rectangle([0, 820, W, H], fill=(150, 110, 80))
    hill_range(img, 760, 40, (160, 110, 90), 2.2)
    return img


def scene_medieval():
    img = gradient((150, 190, 230), (235, 240, 250))
    draw = ImageDraw.Draw(img)
    hill_range(img, 620, 60, (120, 150, 110), 0.9)
    draw = ImageDraw.Draw(img)
    # Крепостна стена с назъбени ръбове и кули
    draw.rectangle([0, 560, W, 800], fill=(140, 125, 110))
    for x in range(0, W, 60):
        draw.rectangle([x, 530, x + 34, 566], fill=(140, 125, 110))
    for tx in (260, 1080, 1640):
        draw.rectangle([tx, 420, tx + 150, 800], fill=(120, 105, 92))
        draw.polygon([(tx - 15, 420), (tx + 165, 420), (tx + 75, 340)], fill=(90, 60, 50))
    # Възрожденски къщи с червени керемиди и прозорци
    rnd = random.Random(7)
    x = 60
    while x < W - 150:
        w = rnd.randint(170, 230)
        h = rnd.randint(200, 280)
        top = 820 - h
        draw.rectangle([x, top, x + w, 820], fill=(235, 220, 190))
        draw.polygon([(x - 10, top), (x + w + 10, top), (x + w // 2, top - 60)],
                     fill=rnd.choice([(180, 80, 50), (160, 62, 40)]))
        for wx in range(x + 25, x + w - 40, 60):
            draw.rectangle([wx, top + 50, wx + 30, top + 100], fill=(90, 70, 60))
        x += w + 30
    draw.rectangle([0, 820, W, H], fill=(105, 90, 75))
    return img


def scene_present():
    img = gradient((110, 150, 210), (250, 195, 150))
    draw = ImageDraw.Draw(img)
    rnd = random.Random(21)
    palette = [(200, 90, 70), (230, 170, 80), (90, 130, 170), (240, 230, 210), (110, 90, 140)]
    x = -20
    while x < W:
        w = rnd.randint(140, 260)
        h = rnd.randint(260, 560)
        color = rnd.choice(palette)
        draw.rectangle([x, 820 - h, x + w, 820], fill=color)
        for wy in range(820 - h + 30, 800, 70):
            for wx in range(x + 20, x + w - 30, 50):
                lit = rnd.random() < 0.35
                draw.rectangle([wx, wy, wx + 22, wy + 34],
                               fill=(255, 225, 150) if lit else (60, 70, 90))
        x += w + rnd.randint(5, 25)
    draw.rectangle([0, 820, W, H], fill=(70, 70, 80))
    for lx in range(80, W, 320):
        draw.rectangle([lx, 900, lx + 10, 1000], fill=(40, 40, 50))
    return img


def scene_outro():
    img = gradient((10, 15, 40), (60, 40, 90))
    rnd = random.Random(3)
    draw = ImageDraw.Draw(img)
    for _ in range(260):
        x, y = rnd.randint(0, W), rnd.randint(0, 600)
        r = rnd.choice([1, 1, 2, 2, 3])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(230, 230, 255))
    img = glow(img, (960, 520), 260, (90, 60, 120), blur=150)
    hill_range(img, 760, 70, (30, 20, 50), 1.1)
    draw = ImageDraw.Draw(img)
    for _ in range(120):
        x, y = rnd.randint(0, W), rnd.randint(720, 900)
        draw.rectangle([x, y, x + 6, y + 6], fill=(255, 200, 120))
    return img


SCENES = {
    "01_intro.jpg": scene_intro,
    "02_antiquity.jpg": scene_antiquity,
    "03_medieval_revival.jpg": scene_medieval,
    "04_present.jpg": scene_present,
    "05_outro.jpg": scene_outro,
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder in SCENES.items():
        builder().save(OUT / name, quality=92)
        print(f"Създаден: {OUT / name}")


if __name__ == "__main__":
    main()
