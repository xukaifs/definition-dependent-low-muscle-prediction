from pathlib import Path
import hashlib

root = Path(__file__).resolve().parent
manifest = root / "SHA256SUMS.txt"
bad = []
checked = 0
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    digest, rel = line.split("  ", 1)
    path = root / rel
    if not path.exists():
        bad.append((rel, "MISSING"))
        continue
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    checked += 1
    if got != digest:
        bad.append((rel, "HASH_MISMATCH"))
if bad:
    print("FAILED")
    for item in bad:
        print(item)
    raise SystemExit(1)
print(f"OK: {checked} files verified.")
