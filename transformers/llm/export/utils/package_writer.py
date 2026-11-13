#
#  package_writer.py
#  MNN
#
#  Created by MNN on 2026/11/09.
#  Copyright © 2018, Alibaba Group Holding Limited
#

"""Write a single-file LLM resource package."""

import os
from pathlib import Path, PureWindowsPath
from runpy import run_path
import shutil
import struct
import tempfile
from zlib import adler32


HEADER = struct.Struct('<7I')     # magic, kind, version, directory size, Adler-32, header_size, required_features
ENTRY = struct.Struct('<IIQ')     # UTF-8 path size, storage (0 inline / 1 file region), size
ALIGNMENT = 16384
DIRECTORY_LIMIT = 64 * 1024 * 1024


def register_file_region(args, path):
    """Mark exported weight data for bounded, on-demand file access."""
    if getattr(args, 'dst_file', None):
        name = Path(path).resolve().relative_to(Path(args.dst_path).resolve()).as_posix()
        args.file_regions.append(name)
    return path


def collect_resources(directory, file_regions=()):
    """Collect a runtime directory; unmarked resources use protected inline storage."""
    root = Path(directory).resolve()
    regions = set(file_regions)
    files = {}
    found = set()
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError(f'Expected a regular resource file or directory: {path}')
        if path.is_dir():
            continue
        name = path.relative_to(root).as_posix()
        if PureWindowsPath(name).anchor or ':' in name or '\\' in name:
            raise ValueError(f'Invalid resource path: {name}')
        found.add(name)
        region = name in regions
        # Converters may emit empty external data for graphs with no weights.
        if region and path.stat().st_size == 0:
            continue
        files[name] = region
    if regions - found:
        raise ValueError('A registered file region is missing from the runtime directory')
    if not 0 < len(files) <= 1024:
        raise ValueError('A package must contain between 1 and 1024 resources')
    return files


def _pack_format(root, files, output):
    entries = bytearray(struct.pack('<I', len(files)))
    regions = []
    index_size = 4 + sum(ENTRY.size + len(k.encode('utf-8')) + 7 for k in files)
    for key, region in files.items():
        path = root / key
        size = path.stat().st_size
        if size <= 0 or output.resolve() == path.resolve():
            raise ValueError(f'Empty resource or output overlaps an input: {key}')
        name = key.encode('utf-8')
        region = region or size > DIRECTORY_LIMIT - index_size
        entries.extend(ENTRY.pack(len(name), int(region), size))
        entries.extend(name)
        if region:
            regions.append((path, size))
        else:
            index_size += size
            entries.extend(bytes(-len(entries) % 8))
            with path.open('rb') as src:
                data = src.read(size + 1)
            if len(data) != size:
                raise ValueError(f'Input changed during export: {key}')
            entries.extend(data)
    if len(entries) > DIRECTORY_LIMIT:
        raise ValueError('Single-file resource directory exceeds 64 MiB')
    return HEADER.pack(430, 1, 3, len(entries), adler32(entries), HEADER.size, 0) + entries, regions, None


def pack_model(directory, output, file_regions=()):
    root = Path(directory).resolve()
    output = Path(output).absolute()
    files = collect_resources(root, file_regions)
    rules = Path(__file__).with_name('pack_format.py')
    pack_format = run_path(str(rules))['pack_format'] if rules.is_file() else _pack_format
    prefix, regions, transform = pack_format(root, files, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.llm-') as tmp:
        bundle = Path(tmp) / 'bundle'
        with bundle.open('wb') as dst:
            dst.write(prefix)
            for path, size in regions:
                padding = bytes(-dst.tell() % ALIGNMENT)
                dst.write(transform(padding, dst.tell()) if transform else padding)
                start = dst.tell()
                if size > (1 << 63) - 1 - start:
                    raise ValueError('Package exceeds signed 64-bit file offsets')
                with path.open('rb') as src:
                    if transform:
                        while True:
                            data = src.read(1024 * 1024)
                            if not data:
                                break
                            dst.write(transform(data, dst.tell()))
                    else:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                if dst.tell() - start != size:
                    raise ValueError(f'Input changed during export: {path}')
        os.replace(bundle, output)
    return str(output)
