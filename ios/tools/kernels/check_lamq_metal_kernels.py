"""Compile the lamq Metal kernels (InklingLamqKernels.swift) the way MLX builds them at run time.

`MLXFast.metalKernel` sources are Swift string literals that MLX compiles on the device the first
time each template instantiation runs, so the app's Xcode build never sees them. This script
rebuilds the exact source MLX generates (`write_signature` in mlx/backend/common/metal_kernel.cpp,
as in PR #8's check_metal_kernels.py, whose generator this mirrors) for every instantiation the
lamq builds need, and compiles each with `xcrun metal`.

    python check_lamq_metal_kernels.py --out build/lamq-metal [--mlx-include <mlx root>] [--compile]

Without --compile it only writes the .metal files (any OS). When PR #8 (`inkling-kernels`) lands,
fold `specs()` into its check_metal_kernels.py instead of keeping two generators.
"""
from __future__ import annotations

import argparse
import itertools
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SWIFT = os.path.join(REPO, "ios", "Vendor", "mlx-swift-lm", "Libraries", "MLXVLM", "Models", "InklingLamqKernels.swift")

_DECL = re.compile(r"^(?:private |fileprivate |internal |public )?let (\w+)\s*=\s*", re.M)


def _swift_unescape(s: str, where: str) -> str:
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        n = s[i + 1]
        if n == "(":
            raise ValueError(f"{where}: string interpolation is not supported")
        if n == "\n":
            i += 2
            continue
        out.append({"\\": "\\", "n": "\n", "t": "\t", "r": "\r", "0": "\0", '"': '"', "'": "'"}[n])
        i += 2
    return "".join(out)


def swift_strings(path: str) -> dict[str, str]:
    """Every top-level `let name = \"\"\"...\"\"\"` multi-line literal, indentation stripped."""
    text = open(path, encoding="utf-8").read()
    values = {}
    for m in _DECL.finditer(text):
        pos = m.end()
        if not text.startswith('"""', pos):
            continue
        body_start = text.index("\n", pos) + 1
        close = text.index('"""', body_start)
        line_start = text.rindex("\n", 0, close) + 1
        indent = text[line_start:close]
        assert indent.strip() == "", f"{m.group(1)}: closing delimiter must be on its own line"
        lines = text[body_start:line_start - 1].split("\n") if line_start > body_start else []
        stripped = []
        for line in lines:
            if line.strip() == "":
                stripped.append("")
            else:
                assert line.startswith(indent), f"{m.group(1)}: under-indented line {line!r}"
                stripped.append(line[len(indent):])
        values[m.group(1)] = _swift_unescape("\n".join(stripped), m.group(1))
    return values


METAL_ATTRIBUTES = [  # metal_kernel.cpp, in its order
    ("dispatch_quadgroups_per_threadgroup", "uint"),
    ("dispatch_simdgroups_per_threadgroup", "uint"),
    ("dispatch_threads_per_threadgroup", "uint3"),
    ("grid_origin", "uint3"),
    ("grid_size", "uint3"),
    ("quadgroup_index_in_threadgroup", "uint"),
    ("quadgroups_per_threadgroup", "uint"),
    ("simdgroup_index_in_threadgroup", "uint"),
    ("simdgroups_per_threadgroup", "uint"),
    ("thread_execution_width", "uint"),
    ("thread_index_in_quadgroup", "uint"),
    ("thread_index_in_simdgroup", "uint"),
    ("thread_index_in_threadgroup", "uint"),
    ("thread_position_in_grid", "uint3"),
    ("thread_position_in_threadgroup", "uint3"),
    ("threadgroup_position_in_grid", "uint3"),
    ("threadgroups_per_grid", "uint3"),
    ("threads_per_grid", "uint3"),
    ("threads_per_simdgroup", "uint"),
    ("threads_per_threadgroup", "uint3"),
]

DEVICE, CONSTANT = "device", "constant"


@dataclass
class Instance:
    kernel: str
    source: str
    header: str | None
    inputs: list[tuple[str, str, str]]
    outputs: list[tuple[str, str]]
    template: list[tuple[str, object]] = field(default_factory=list)
    label: str = ""


def _tv(v) -> str:
    return ("1" if v else "0") if isinstance(v, bool) else str(v)


def _tp(v) -> str:
    return "bool" if isinstance(v, bool) else ("int" if isinstance(v, int) else "typename")


def generate(inst: Instance, strings: dict[str, str], prelude: str) -> str:
    source = strings[inst.source]
    header = strings[inst.header] if inst.header else ""
    tdef = "<" + ", ".join(_tv(v) for _, v in inst.template) + ">"
    func = "custom_kernel_" + inst.kernel
    if inst.template:
        func += "_" + re.sub(r"[<>\-]|, ", "_", tdef)[:-1]
    for _, t, loc in inst.inputs:
        func += "_" + t + ("c" if loc == CONSTANT else "")
    for _, t in inst.outputs:
        func += "_" + t
    func = re.sub(r"\W", "_", func)
    src = [prelude, header]
    if inst.template:
        src.append("template <" + ", ".join(f"{_tp(v)} {n}" for n, v in inst.template) + ">\n")
    src.append(f"[[kernel]] void {func}(\n")
    params, index = [], 0
    for name, t, loc in inst.inputs:
        params.append(f"  const {loc} {t}* {name} [[buffer({index})]]")
        index += 1
        for suffix, decl in (("_shape", "const constant int*"), ("_strides", "const constant int64_t*"), ("_ndim", "const constant int&")):
            if name + suffix in source:
                params.append(f"  {decl} {name}{suffix} [[buffer({index})]]")
                index += 1
    for name, t in inst.outputs:
        params.append(f"  device {t}* {name} [[buffer({index})]]")
        index += 1
    for attr, t in METAL_ATTRIBUTES:
        if attr in source:
            params.append(f"  {t} {attr} [[{attr}]]")
    src.append(",\n".join(params) + ") {\n")
    src.append(source)
    src.append("\n}\n")
    if inst.template:
        full = func + tdef
        src.append(f'\ntemplate [[host_name("{func}")]] [[kernel]] decltype({full}) {full};\n')
    return "".join(src)


F16, F32, U8, U32 = "float16_t", "float", "uint8_t", "uint32_t"
# fmt -> (k, g, log2 K, hashed)
TRELLIS = {"ht1": (1, 4, 9, True), "ht2": (2, 4, 9, True), "ht3": (3, 2, 10, True),
           "rtr1": (1, 1, 12, False), "rtr2": (2, 1, 12, False), "rtr3": (3, 1, 12, False), "rtr4": (4, 1, 12, False)}
# Input lengths of the Inkling linears: 2304 (q/k/v/rel/gate, gu, embed/head), 1152 (dn), 3072 (o).
NS = (2304, 1152, 3072)


def specs() -> list[Instance]:
    out = []
    for (fmt, (k, g, logk, hashed)), n in itertools.product(TRELLIS.items(), NS):
        tmpl = [("KB", k), ("G", g), ("LOGK", logk), ("HASHED", hashed), ("N", n)]
        for sel_loc in (DEVICE, CONSTANT):   # MLX puts arrays of < 8 elements in constant space
            for t in (F32, F16):
                out.append(Instance(
                    "lamq_trellis_dequant", "lamqTrellisDequantSource", "lamqHeader",
                    [("codes", U32, DEVICE), ("scales", F16, DEVICE), ("lut", F16, DEVICE), ("locs", U32, sel_loc), ("params", U32, CONSTANT)],
                    [("out", t)], tmpl + [("T", t)], f"dequant-{fmt}-n{n}-{t}-{sel_loc}"))
            out.append(Instance(
                "lamq_trellis_matvec", "lamqTrellisMatvecSource", "lamqHeader",
                [("x", F32, DEVICE), ("codes", U32, DEVICE), ("scales", F16, DEVICE), ("lut", F16, DEVICE), ("sel", U32, sel_loc), ("params", U32, CONSTANT)],
                [("y", F32)], tmpl + [("NSG", 8)], f"matvec-{fmt}-n{n}-{sel_loc}"))
    for k, n in itertools.product((6, 8), NS):
        for t in (F32, F16):
            for loc in (DEVICE, CONSTANT):
                out.append(Instance(
                    "lamq_lm_dequant", "lamqLMDequantSource", None,
                    [("codes", U8, DEVICE), ("scales", F16, DEVICE), ("levels", F32, DEVICE), ("rows_sel", U32, loc)],
                    [("out", t)], [("KB", k), ("N", n), ("T", t)], f"lm-dequant-k{k}-n{n}-{t}-{loc}"))
        out.append(Instance(
            "lamq_lm_matvec", "lamqLMMatvecSource", None,
            [("x", F32, DEVICE), ("codes", U8, DEVICE), ("scales", F16, DEVICE), ("levels", F32, DEVICE), ("params", U32, CONSTANT)],
            [("y", F32)], [("KB", k), ("N", n), ("NSG", 8)], f"lm-matvec-k{k}-n{n}"))
    return out


FALLBACK_PRELUDE = """#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--swift", default=SWIFT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlx-include")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--sdk", default="iphoneos")
    ap.add_argument("--std", default="metal3.1")
    args = ap.parse_args()

    strings = swift_strings(args.swift)
    swift_text = open(args.swift, encoding="utf-8").read()
    prelude, include = FALLBACK_PRELUDE, []
    if args.mlx_include:
        utils = os.path.join(args.mlx_include, "mlx", "backend", "metal", "kernels", "utils.h")
        if not os.path.exists(utils):
            print(f"error: {utils} not found", file=sys.stderr)
            return 2
        prelude, include = '#include "mlx/backend/metal/kernels/utils.h"\n', ["-I", args.mlx_include]
    os.makedirs(args.out, exist_ok=True)
    failures, files = [], []
    for inst in specs():
        if f'name: "{inst.kernel}"' not in swift_text:
            failures.append((inst.label, f"no metalKernel named {inst.kernel}"))
            continue
        missing = [n for n, *_ in inst.inputs + inst.outputs if f'"{n}"' not in swift_text]
        if missing:
            failures.append((inst.label, f"argument names not in the Swift file: {missing}"))
            continue
        path = os.path.join(args.out, inst.label + ".metal")
        with open(path, "w", encoding="utf-8") as f:
            f.write(generate(inst, strings, prelude))
        files.append((inst.label, path))
    print(f"generated {len(files)} kernel sources in {args.out}")
    if args.compile:
        for label, path in files:
            r = subprocess.run(["xcrun", "-sdk", args.sdk, "metal", f"-std={args.std}", *include, "-c", path,
                                "-o", path[:-6] + ".air"], capture_output=True, text=True)
            if r.returncode != 0:
                failures.append((label, r.stderr.strip()))
            else:
                warnings = [l for l in r.stderr.splitlines() if "warning:" in l]
                if warnings:
                    print(f"{label}: {len(warnings)} warning(s)\n  " + "\n  ".join(warnings[:5]))
        print(f"compiled {len(files) - len(failures)}/{len(files)} without errors")
    for label, msg in failures:
        print(f"::error title=lamq Metal kernel {label}::{msg.splitlines()[0] if msg else ''}")
        print(msg)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
