#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 [-p python_version] [-t] [-c]"
    echo "  -p  select a pyenv Python version without changing the global version"
    echo "  -t  include train API wrappers"
    echo "  -c  check whether the resources need updating"
}

train_api=false
check=false
while getopts "p:tc" opt; do
    case "$opt" in
        p) export PYENV_VERSION="$OPTARG" ;;
        t) train_api=true ;;
        c) check=true ;;
        *) usage; exit 2 ;;
    esac
done

root="${MNN_WRAPPER_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
comment_script="${MNN_WRAPPER_COMMENT_SCRIPT:-$root/rm_comments.py}"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/mnn-py.XXXXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT
cp -R "$root/pip_package/MNN" "$tmp_dir/MNN"
python3 "$comment_script" "$tmp_dir/MNN"

rm -rf "$tmp_dir/MNN/tools" "$tmp_dir/MNN/audio"
if [[ "$train_api" == false ]]; then
    rm -rf "$tmp_dir/MNN/data" "$tmp_dir/MNN/optim"
fi
python3 - "$tmp_dir/MNN/__init__.py" "$train_api" <<'PY'
import sys
path, train = sys.argv[1:]
excluded = {'tools', 'audio'} | (set() if train == 'true' else {'data', 'optim'})
with open(path) as f:
    lines = f.readlines()
with open(path, 'w') as f:
    f.writelines(line for line in lines if line.strip() not in
                 {'from . import ' + name for name in excluded})
PY
find "$tmp_dir/MNN" -type d -name __pycache__ -prune -exec rm -rf {} +

python2 - "$tmp_dir/MNN" <<'PY'
import compileall
import sys
if sys.version_info[:2] != (2, 7):
    sys.exit('Python 2.7 is required')
if not compileall.compile_dir(sys.argv[1], ddir='/tmp/mnn_py/MNN', force=True):
    sys.exit('Python wrapper compilation failed')
PY
find "$tmp_dir/MNN" -type f -name '*.py' -delete

python3 - "$tmp_dir/MNN" "$root/iOS/MNNPyBridge/lib/MNN" \
    "$root/android/src/main/assets/MNN.zip" "$check" <<'PY'
import os
import shutil
import sys
import zipfile
from pathlib import Path

generated, ios, archive = map(Path, sys.argv[1:4])
check = sys.argv[4] == 'true'

def files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*.pyc')}

def equivalent(a, b):
    return len(a) >= 8 and len(b) >= 8 and a[:4] == b[:4] and a[8:] == b[8:]

new = files(generated)
old = files(ios)
if not new:
    sys.exit('No Python wrapper bytecode was generated')
try:
    with zipfile.ZipFile(archive) as z:
        entries = [n for n in z.namelist() if not n.endswith('/')]
        zipped = {n[4:]: z.read(n) for n in entries if n.startswith('MNN/')}
        zip_valid = len(entries) == len(zipped) and zipped == old and z.testzip() is None
except (OSError, zipfile.BadZipFile):
    zip_valid = False

needs_update = (new.keys() != old.keys() or
                any(not equivalent(new[name], old[name]) for name in new.keys() & old.keys()) or
                not zip_valid)
if not needs_update:
    print('need not update')
    sys.exit(0)
if check:
    print('need update')
    sys.exit(1)

for name in old.keys() - new.keys():
    (ios / name).unlink()
for name, data in new.items():
    path = ios / name
    if name not in old or not equivalent(data, old[name]):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated / name, path)
archive.parent.mkdir(parents=True, exist_ok=True)
temporary = archive.with_name(archive.name + '.tmp')
try:
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        directories = {'MNN/'}
        for name in new:
            parent = Path(name).parent
            while str(parent) != '.':
                directories.add('MNN/' + str(parent) + '/')
                parent = parent.parent
        for directory in sorted(directories):
            z.writestr(directory, b'')
        for name in sorted(new):
            z.write(ios / name, 'MNN/' + name)
    os.replace(temporary, archive)
finally:
    if temporary.exists():
        temporary.unlink()
print('updated Python wrapper assets')
PY
