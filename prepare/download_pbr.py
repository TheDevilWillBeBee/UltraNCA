"""Download the PBR texture dataset (as in Cells2Pixels/scripts/download_data.py) and write the albedo maps as a flat
folder of training targets: data/pbr_textures/<texture name>.jpg (73 textures).
python prepare/download_pbr.py [--out data/pbr_textures]
"""
import os, argparse, shutil, tempfile, zipfile
import gdown

URL = 'https://drive.google.com/uc?id=1TsA3fyr1OU_1C4Rk5rtBp-Zzr_54Bofn'  # PBR Texture dataset (Cells2Pixels)

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'pbr_textures'))
out = ap.parse_args().out
if os.path.isdir(out) and os.listdir(out):
    raise SystemExit(f'{out} exists and is not empty; remove it to download again')
os.makedirs(out, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    zip_path = os.path.join(tmp, 'pbr_textures.zip')
    gdown.download(URL, zip_path, quiet=False)
    with zipfile.ZipFile(zip_path) as z:
        names = sorted({n.split('/')[0] for n in z.namelist() if n.endswith('/albedo.jpg')})
        for name in names:
            with z.open(f'{name}/albedo.jpg') as src, open(os.path.join(out, f'{name}.jpg'), 'wb') as dst:
                shutil.copyfileobj(src, dst)
print(f'wrote {len(names)} albedo textures to {out}')
