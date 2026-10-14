"""Download 64 Noto emojis (128 x 128 RGBA PNG) as growing-NCA targets: data/emoji/<codepoint>.png.
python prepare/download_emoji.py [--out data/emoji]
"""
import os, argparse, urllib.request

URL = 'https://raw.githubusercontent.com/googlefonts/noto-emoji/v2.047/png/128/emoji_u{:x}.png'
EMOJIS = ('🦎😀💥👁🐠🦋🐞🕸🥨🎄🐙🦀🐢🐸🦊🐶🐱🐭🐰🐻🐼🐨🐯🦁🐮🐷🐵🐔🐧🐦🦆🦉🦇🐺🐗🐴🦄🐝🐛🐌'
          '🐜🦂🐍🦖🐳🐬🦈🐊🦓🦒🌵🌻🍄🍎🍋🍉🍇🍓🍕🍔🍩🎈🚀⚽')

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'emoji'))
out = ap.parse_args().out
os.makedirs(out, exist_ok=True)
assert len(EMOJIS) == 64, len(EMOJIS)
for e in EMOJIS:
    path = os.path.join(out, f'{ord(e):x}.png')
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL.format(ord(e)), path)
print(f'{len(EMOJIS)} emojis in {out}')
