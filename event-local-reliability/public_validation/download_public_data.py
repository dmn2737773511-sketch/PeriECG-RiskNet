"""Download the exact public PhysioNet inputs in FROZEN_PLAN.md."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, hashlib, json, time

ROOT = Path(__file__).resolve().parent
BASE = 'https://physionet.org/files/'
EXPECTED = {(r['dataset'], r['file']): r for r in json.loads((ROOT / 'download_manifest.json').read_text())}

def fetch(item):
    dataset, name = item
    path = ROOT / 'raw' / dataset / name
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f'{BASE}{dataset}/1.0.0/{name}'
    if not path.exists():
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                path.write_bytes(data)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)
    data = path.read_bytes()
    expected = EXPECTED[item]
    if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
        raise ValueError(f'Input hash differs from frozen release: {dataset}/{name}')
    return {'dataset': dataset, 'file': name, 'url': url,
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def main():
    manifest = [fetch(('mitdb', 'RECORDS'))]
    records = (ROOT / 'raw' / 'mitdb' / 'RECORDS').read_text().split()
    if len(records) != 48:
        raise ValueError(f'Expected 48 ECG records, found {len(records)}')
    jobs = [('mitdb', f'{r}.{ext}') for r in records for ext in ('hea', 'dat', 'atr')]
    jobs += [('nstdb', f'{r}.{ext}') for r in ('bw','ma','em') for ext in ('hea','dat')]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, future in enumerate(as_completed([pool.submit(fetch, job) for job in jobs])):
            manifest.append(future.result())
            if (i+1) % 15 == 0:
                print(f'downloaded {i+1}/{len(jobs)} files', flush=True)
    manifest.sort(key=lambda row: (row['dataset'],row['file']))
    (ROOT / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Download complete: {len(manifest)} files, {sum(x["bytes"] for x in manifest)} bytes', flush=True)

if __name__ == '__main__':
    main()
