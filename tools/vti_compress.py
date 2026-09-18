"""
Compress a run's VTI frames IN PLACE, keeping them ParaView-native.

    python tools/vti_compress.py --run-dir domains/greenland/inverse/projection_CESM2-WACCM_ssp585
    python tools/vti_compress.py --run-dir ... --workers 6 --level 9

Each frame is rewritten with

  * velocities and SMB zeroed where the active-set mask is 1 (ice-free: the
    2/3 of the grid that holds solver noise and makes those arrays
    incompressible),
  * values rounded to the physical precisions of tools/vti_to_nc.py
    (1 cm, 0.01 m/yr, 1 mm/yr, 1e-4 for the flags),
  * VTK's compressed appended-data layout with vtkLZ4DataCompressor
    (`header_type="UInt32"`, 32 KiB blocks, raw LZ4 blocks),

which ParaView (5.5+) reads directly: the .pvd is untouched, the time
slider still shows model years, and only the arrays you tick are
decompressed (LZ4 inflates at GB/s, so a frame loads faster than the raw
326 MB file did). A 1 km frame drops to ~70 MB (x4.6; the LZ4 level barely
matters, zlib would give ~50 MB but inflates 5x slower). The array layout (names,
components, point data, the ascii TimeValue) is preserved, so every reader
of the raw files that understands the compressed layout (ismip_exporter,
analysis/basin_mass_balance.py via `read_vti`) keeps working.

Safety: the new frame is written to a temporary file, decoded back and
compared with the rounded arrays bit for bit, and only then moved over the
original. Frames already carrying a `compressor` attribute are skipped.
"""
import argparse
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import lz4.block
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vti_to_nc import MASK_OFF_ICE, PRECISION, VECTORS  # noqa: E402

BLOCK = 32768                               # vtkXMLWriter's default BlockSize (bytes)
HEADER = np.uint32
COMPRESSOR = "vtkLZ4DataCompressor"


def parse(path):
    """XML head (bytes, up to and including the '_' that opens the appended
    data), the payload bytes, and the array table [(name, ncomp, offset)]."""
    data = Path(path).read_bytes()
    i = data.index(b"<AppendedData")
    j = data.index(b"_", i) + 1
    head = data[:j]
    xml = head.decode("utf-8", "ignore")
    arrays = [(m.group(1), int(m.group(2)), int(m.group(3))) for m in
              re.finditer(r'Name="(\w+)" NumberOfComponents="(\d+)" format="appended" offset="(\d+)"', xml)]
    ext = [int(v) for v in re.search(r'WholeExtent="([^"]+)"', xml).group(1).split()]
    npts = (ext[1] - ext[0] + 1) * (ext[3] - ext[2] + 1) * (ext[5] - ext[4] + 1)
    return head, data[j:], arrays, npts


def read_raw_array(payload, offset, ncomp, npts, compressed, header=HEADER):
    """One appended Float32 array, raw or compressed layout -> (npts, ncomp)."""
    hs = np.dtype(header).itemsize
    if not compressed:
        n = int(np.frombuffer(payload[offset:offset + hs], header)[0])
        a = np.frombuffer(payload[offset + hs:offset + hs + n], np.float32)
    else:
        nblk, bsize, last = (int(v) for v in np.frombuffer(payload[offset:offset + 3 * hs], header))
        sizes = np.frombuffer(payload[offset + 3 * hs:offset + (3 + nblk) * hs], header)
        pos = offset + (3 + nblk) * hs
        chunks = []
        for k, cs in enumerate(sizes):
            usize = bsize if (k < nblk - 1 or last == 0) else last
            chunks.append(lz4.block.decompress(payload[pos:pos + int(cs)], uncompressed_size=usize))
            pos += int(cs)
        a = np.frombuffer(b"".join(chunks), np.float32)
    return a.reshape(npts, ncomp) if ncomp > 1 else a


def compress_array(raw: bytes, level: int) -> bytes:
    """VTK compressed appended layout: [nblocks, blocksize, last, sizes...] + LZ4 blocks."""
    n = len(raw)
    nfull, last = divmod(n, BLOCK)
    nblk = nfull + (1 if last else 0)
    blocks = [raw[k * BLOCK:(k + 1) * BLOCK] for k in range(nblk)]
    comp = [lz4.block.compress(b, mode="high_compression", compression=level, store_size=False) for b in blocks]
    hdr = np.array([nblk, BLOCK, last] + [len(c) for c in comp], dtype=HEADER).tobytes()
    return hdr + b"".join(comp)


def quantize(name, a, ice, ncomp):
    """Mask off-ice and round, per component of vector arrays."""
    if ncomp > 1:
        comps = VECTORS.get(name, tuple(f"{name}{k}" for k in range(ncomp)))
        return np.stack([quantize(c, a[:, k], ice, 1) for k, c in enumerate(comps)], axis=1)
    if name in MASK_OFF_ICE and ice is not None:
        a = np.where(ice, a, 0.0)
    p = PRECISION.get(name)
    return (np.round(a / p) * p).astype(np.float32) if p else a.astype(np.float32)


def compress_file(path, level=1, do_round=True):
    path = Path(path)
    head, payload, arrays, npts = parse(path)
    xml = head.decode("utf-8", "ignore")
    if "compressor=" in xml:
        return path, path.stat().st_size, path.stat().st_size, "skipped (already compressed)"
    old_size = path.stat().st_size
    fields = {n: read_raw_array(payload, off, nc, npts, compressed=False) for n, nc, off in arrays}
    ice = (fields["mask"] < 0.5) if ("mask" in fields and do_round) else None
    blobs, offsets, pos = [], {}, 0
    for n, nc, _ in arrays:
        a = quantize(n, fields[n], ice, nc) if do_round else fields[n].astype(np.float32)
        b = compress_array(np.ascontiguousarray(a).tobytes(), level)
        offsets[n] = pos
        blobs.append(b)
        pos += len(b)
    # new head: declare the compressor and header type on the VTKFile tag, renumber offsets
    xml = re.sub(r'<VTKFile([^>]*)>', lambda m: "<VTKFile" + re.sub(r'\s+(header_type|compressor)="[^"]*"', "", m.group(1))
                 + f' header_type="UInt32" compressor="{COMPRESSOR}">', xml, count=1)
    xml = re.sub(r'(Name="(\w+)" NumberOfComponents="\d+" format="appended" offset=")(\d+)"',
                 lambda m: f'{m.group(1)}{offsets[m.group(2)]}"', xml)
    tmp = path.with_suffix(".vti.tmp")
    with open(tmp, "wb") as f:
        f.write(xml.encode("utf-8"))
        for b in blobs:
            f.write(b)
        f.write(b"\n  </AppendedData>\n</VTKFile>\n")
    # verify by decoding the new file
    head2, payload2, arrays2, npts2 = parse(tmp)
    assert npts2 == npts and [a[:2] for a in arrays2] == [a[:2] for a in arrays], "array table changed"
    for n, nc, off in arrays2:
        want = quantize(n, fields[n], ice, nc) if do_round else fields[n].astype(np.float32)
        got = read_raw_array(payload2, off, nc, npts, compressed=True)
        if not np.array_equal(got.reshape(want.shape), want):
            tmp.unlink()
            raise RuntimeError(f"{path.name}: round trip mismatch in {n}")
    os.replace(tmp, path)
    return path, old_size, path.stat().st_size, "ok"


def _job(args):
    try:
        return compress_file(*args)
    except Exception as e:                   # report, keep going; the original is untouched
        return Path(args[0]), 0, 0, f"FAILED: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--level", type=int, default=1, help="LZ4 HC level (1 .. 12; the size is the same to 1%, so 1)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-round", action="store_true", help="compress only; keep every bit and the off-ice noise")
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames (a trial)")
    a = ap.parse_args()
    vti_dir = Path(a.run_dir) / "vti"
    files = sorted(vti_dir.glob("*.vti"))
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f"no .vti files in {vti_dir}")
    print(f"{len(files)} frames in {vti_dir}, LZ4 HC level {a.level}, {a.workers} workers", flush=True)
    tic = time.time()
    tot_old = tot_new = 0
    n_done = 0
    with Pool(a.workers) as pool:
        for path, old, new, status in pool.imap_unordered(_job, [(str(p), a.level, not a.no_round) for p in files]):
            tot_old += old; tot_new += new; n_done += 1
            if status != "ok" or n_done % 20 == 0 or n_done == len(files):
                rate = f"x{old / new:.1f}" if new else ""
                print(f"  [{n_done}/{len(files)}] {path.name}: {old / 1e6:.0f} -> {new / 1e6:.0f} MB {rate} {status}  "
                      f"({time.time() - tic:.0f} s)", flush=True)
    print(f"done: {tot_old / 1e9:.1f} GB -> {tot_new / 1e9:.1f} GB (x{tot_old / max(tot_new, 1):.1f}) in {time.time() - tic:.0f} s")


if __name__ == "__main__":
    main()
