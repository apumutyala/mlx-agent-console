"""Compile the Metal kernels that Swift code builds at runtime with ``MLXFast.metalKernel``.

The app's Xcode build never sees these kernels: they are Swift string literals that MLX
compiles on the device, the first time each template instantiation runs. A typo there only
shows up as a runtime error on the iPad. This script rebuilds the exact source MLX would
generate (``write_signature`` in mlx/backend/common/metal_kernel.cpp at the mlx commit
mlx-swift 0.33.0 bundles) for the instantiations listed in ``SPECS`` and compiles each with
``xcrun metal``.

    python check_metal_kernels.py --out build/metal-check \
        [--mlx-include <mlx checkout root>] [--compile]

Without ``--compile`` it only writes the ``.metal`` files (works on any OS). With
``--mlx-include`` the prelude is MLX's own ``kernels/utils.h`` (what ``metal::utils()`` holds);
otherwise a minimal stdlib + bfloat prelude.

To cover another kernel, add a spec: its Swift string constants, its inputs as MLX would pass
them (dtype, and whether MLX puts it in the ``constant`` address space: arrays of fewer than 8
elements; ``scalar`` for 0-d arrays) and the template values to instantiate.
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
INKLING_KERNELS = os.path.join(
    REPO, "ios", "Vendor", "mlx-swift-lm", "Libraries", "MLXVLM", "Models", "Inkling2Kernels.swift")

# ------------------------------------------------------------- Swift string constants

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
        if n == "u":
            j = s.index("}", i)
            out.append(chr(int(s[i + 3:j], 16)))
            i = j + 1
            continue
        if n == "\n":  # line continuation
            i += 2
            continue
        out.append({"\\": "\\", "n": "\n", "t": "\t", "r": "\r", "0": "\0", '"': '"', "'": "'"}[n])
        i += 2
    return "".join(out)


def _multiline(text: str, start: int, where: str) -> tuple[str, int]:
    """The value of the multi-line literal whose opening ``\"\"\"`` is at ``start``."""
    body_start = text.index("\n", start) + 1
    close = text.index('"""', body_start)
    closing_line_start = text.rindex("\n", 0, close) + 1
    indent = text[closing_line_start:close]
    assert indent.strip() == "", f"{where}: closing delimiter must be on its own line"
    lines = text[body_start:closing_line_start - 1].split("\n") if closing_line_start > body_start else []
    stripped = []
    for line in lines:
        if line.strip() == "":
            stripped.append("")
        else:
            assert line.startswith(indent), f"{where}: line under-indented: {line!r}"
            stripped.append(line[len(indent):])
    return _swift_unescape("\n".join(stripped), where), close + 3


def swift_strings(path: str) -> dict[str, str]:
    """Every ``let name = <literal> (+ <literal or name>)*`` whose parts are multi-line
    string literals or earlier such constants."""
    text = open(path, encoding="utf-8").read()
    values: dict[str, str] = {}
    for m in _DECL.finditer(text):
        name, pos = m.group(1), m.end()
        parts: list[str] = []
        ok = True
        while True:
            while text[pos] in " \t\n":
                pos += 1
            if text.startswith('"""', pos):
                value, pos = _multiline(text, pos, f"{path}:{name}")
                parts.append(value)
            else:
                ident = re.match(r"\w+", text[pos:])
                if not ident or ident.group(0) not in values:
                    ok = False
                    break
                parts.append(values[ident.group(0)])
                pos += len(ident.group(0))
            after = re.match(r"[ \t]*\+", text[pos:])
            if not after:
                break
            pos += after.end()
        if ok and parts:
            values[name] = "".join(parts)
    return values


# ---------------------------------------------------------- MLX's generated signature

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

DEVICE, CONSTANT, SCALAR = "device", "constant", "scalar"


@dataclass
class Instance:
    kernel: str            # the `name:` passed to metalKernel
    source: str            # Swift constant holding the body
    header: str | None     # Swift constant holding the header, if any
    inputs: list[tuple[str, str, str]]   # (name, metal type, DEVICE | CONSTANT | SCALAR)
    outputs: list[tuple[str, str]]       # (name, metal type)
    template: list[tuple[str, object]] = field(default_factory=list)  # int | bool | metal type
    label: str = ""


def _template_value(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def _template_param(v) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    return "typename"


def generate(inst: Instance, strings: dict[str, str], prelude: str) -> str:
    source = strings[inst.source]
    header = strings[inst.header] if inst.header else ""
    tdef = "<" + ", ".join(_template_value(v) for _, v in inst.template) + ">"
    func = "custom_kernel_" + inst.kernel
    if inst.template:
        func += "_" + re.sub(r"[<>\-]|, ", "_", tdef)[:-1]
    for _, t, loc in inst.inputs:
        func += "_" + t + ("s" if loc == SCALAR else "c" if loc == CONSTANT else "")
    for _, t in inst.outputs:
        func += "_" + t
    func = re.sub(r"\W", "_", func)

    src = [prelude, header]
    if inst.template:
        src.append("template <" + ", ".join(
            f"{_template_param(v)} {n}" for n, v in inst.template) + ">\n")
    src.append(f"[[kernel]] void {func}(\n")
    params, index = [], 0
    for name, t, loc in inst.inputs:
        space = CONSTANT if loc in (CONSTANT, SCALAR) else DEVICE
        ref = "&" if loc == SCALAR else "*"
        params.append(f"  const {space} {t}{ref} {name} [[buffer({index})]]")
        index += 1
        if loc != SCALAR:
            for suffix, decl in (("_shape", "const constant int*"),
                                 ("_strides", "const constant int64_t*"),
                                 ("_ndim", "const constant int&")):
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


# --------------------------------------------------------------------------- specs

BF16, F16, F32, U8, U32 = "bfloat16_t", "float16_t", "float", "uint8_t", "uint32_t"

# Inkling 2 Mini: hidden 2304, intermediate 1152, 8 of 128 experts per token.
D, I, KE = 2304, 1152, 8


def _expert_formats():
    """(label, BITS, GS, AFFINE, HAS_GS, scales type) of every format the fused kernels take."""
    for bits, gs in itertools.product((2, 3, 4, 8), (32, 64, 128)):
        yield f"affine{bits}-gs{gs}", bits, gs, True, False, BF16
    yield "nvfp4", 4, 16, False, True, U8
    yield "mxfp4", 4, 32, False, False, U8
    yield "mxfp8", 8, 32, False, False, U8


def _expert_inputs(scale_t, affine, has_gs, index_loc):
    return [
        ("w", U32, DEVICE),
        ("scales", scale_t, DEVICE),
        ("biases", BF16 if affine else F32, DEVICE if affine else CONSTANT),
        ("global_scale", F32, DEVICE if has_gs else CONSTANT),
        ("indices", U32, index_loc),
    ]


def specs() -> list[Instance]:
    out = []
    for (label, bits, gs, affine, has_gs, scale_t), t in itertools.product(
            _expert_formats(), (BF16, F16)):
        # One token with 8 experts passes `indices` from device memory; MLX would use the
        # constant space below 8 elements, so cover both.
        for index_loc in (DEVICE, CONSTANT):
            ex = _expert_inputs(scale_t, affine, has_gs, index_loc)
            common = [("T", t), ("BITS", bits), ("GS", gs), ("AFFINE", affine), ("HAS_GS", has_gs)]
            tag = f"{label}-{t}-{index_loc}"
            out.append(Instance(
                kernel="inkling2_expert_gate_up", source="expertGateUpSource",
                header="expertHeader",
                inputs=[("x", t, DEVICE)] + ex + [("caps", F32, CONSTANT)],
                outputs=[("act", t)],
                template=common + [("DIN", D), ("I", I), ("KE", KE), ("NSG", 4)],
                label=f"gate_up-{tag}"))
            out.append(Instance(
                kernel="inkling2_expert_down", source="expertDownSource", header="expertHeader",
                inputs=[("act", t, DEVICE)] + ex + [
                    ("weights", F32, index_loc), ("residual", F32, DEVICE),
                    ("state", t, DEVICE), ("conv_weight", t, DEVICE),
                    ("divisor", F32, CONSTANT)],
                outputs=[("hidden", t), ("state_out", t)],
                template=common + [("K", I), ("D", D), ("KE", KE), ("ROWS", 2)],
                label=f"down-{tag}"))
    for t in (BF16, F16):
        out.append(Instance(
            kernel="inkling2_expert_epilogue", source="expertEpilogueSource", header=None,
            inputs=[("outputs", t, DEVICE), ("weights", F32, DEVICE), ("residual", F32, DEVICE),
                    ("state", t, DEVICE), ("conv_weight", t, DEVICE), ("divisor", F32, CONSTANT)],
            outputs=[("hidden", t), ("state_out", t)],
            template=[("T", t), ("D", D), ("KE", KE)], label=f"epilogue-{t}"))
    return out


# ---------------------------------------------------------------------------- main

FALLBACK_PRELUDE = """#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--swift", default=INKLING_KERNELS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlx-include", help="root of an mlx checkout (has mlx/backend/metal/kernels)")
    ap.add_argument("--compile", action="store_true", help="run xcrun metal on every file")
    ap.add_argument("--sdk", default="iphoneos")
    ap.add_argument("--std", default="metal3.1", help="Metal language version (bfloat needs 3.1)")
    args = ap.parse_args()

    strings = swift_strings(args.swift)
    swift_text = open(args.swift, encoding="utf-8").read()
    prelude = FALLBACK_PRELUDE
    include_flags: list[str] = []
    if args.mlx_include:
        utils = os.path.join(args.mlx_include, "mlx", "backend", "metal", "kernels", "utils.h")
        if not os.path.exists(utils):
            print(f"error: {utils} not found", file=sys.stderr)
            return 2
        prelude = '#include "mlx/backend/metal/kernels/utils.h"\n'
        include_flags = ["-I", args.mlx_include]

    os.makedirs(args.out, exist_ok=True)
    failures, files = [], []
    for inst in specs():
        if f'name: "{inst.kernel}"' not in swift_text:
            failures.append((inst.label, f"no metalKernel named {inst.kernel} in {args.swift}"))
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
            cmd = ["xcrun", "-sdk", args.sdk, "metal", f"-std={args.std}", *include_flags, "-c", path,
                   "-o", path[:-len(".metal")] + ".air"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            warnings = [l for l in r.stderr.splitlines() if "warning:" in l and path in l]
            if r.returncode != 0:
                failures.append((label, r.stderr.strip()))
            elif warnings:
                print(f"{label}: {len(warnings)} warning(s)\n  " + "\n  ".join(warnings[:5]))
        print(f"compiled {len(files) - len(failures)}/{len(files)} without errors")

    for label, msg in failures:
        print(f"::error title=Metal kernel {label}::{msg.splitlines()[0] if msg else ''}")
        print(msg)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
