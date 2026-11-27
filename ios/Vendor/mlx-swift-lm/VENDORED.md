# Vendored mlx-swift-lm

Source: https://github.com/aleroot/mlx-swift-lm at `5507b3b3127b015ce27a3f5013e4fb6bde04cb11`
(head of branch `inkling2mini`, i.e. open PR https://github.com/ml-explore/mlx-swift-lm/pull/821
"Add Inkling 2 Mini and Small"). That commit is mlx-swift-lm tag `3.33.0` plus 6 upstream
commits plus the 16 PR commits.

Copied: `Libraries/`, `Package.swift`, `LICENSE`, `ACKNOWLEDGMENTS.md`. Not copied: `Tests/`,
`IntegrationTesting/`, `skills/`, `tools/`, `scripts/`, `.github/`.

## Local changes

1. `Package.swift`: removed the five `.testTarget` entries (their `Tests/` sources are not vendored).
2. `Libraries/MLXVLM/Models/Inkling2.swift`, `prepareCheckpoint`: strip the `model.` scope from the
   per-layer `quantization` keys as well as from the tensor names. Mixed-precision conversions such
   as `LocallyInferred/Inkling-2-Mini-MLX-3.1bpw-DWQ` and `roman220220/Inkling-2-Mini-gptq-mlx-jang*`
   declare their 6/8-bit attention, embeddings, etc. as `model.language_model.…` over a 2- or 3-bit
   default; without the rename those layers were built at the default bits and strict loading failed
   on shape mismatches. Uniform conversions (`mlx-community/Inkling-2-Mini-4bit`) are unaffected.
3. `Libraries/MLXVLM/Models/Inkling2.swift`: `excludedCheckpointNamespaces = ["dspark", "mtp"]`, so a
   DSpark draft that a conversion's index places beside the weights (roman220220's
   `model-mtp.safetensors`) is neither read nor rejected by strict validation.

### Kernel patches (`inkling-kernels`, labelled K so they don't collide with other branches' numbering)

- **K1** `Libraries/MLXVLM/Models/Inkling2Kernels.swift`: 2- and 3-bit affine experts in the fused decode
   expert kernels. `inkling2_unit_dot` gains a 2-bit branch (16 codes in one `uint`) and a 3-bit branch
   (two 24-bit packs of 8 codes, read bytewise because the 6-byte unit is only 2-byte aligned), in
   MLX's affine layout (mlx `quantize`, codes from the lowest bit up). `supportsFusedDecode` accepts
   bits 2/3/4/8 for affine (fp modes stay 4/8). Kernel signatures, grid and indexing are unchanged; the
   bit width is still a template argument per projection, so gate/up and down may differ. Self-contained
   and in the PR's style, so it can be proposed upstream once measured. Not marked inline, to keep
   the diff upstreamable.
- **K2** (local only, marked `[agent-console patch K2]`): the new file
  `Libraries/MLXVLM/Models/Inkling2KernelSelfCheck.swift` holds `Inkling2FusedDecodeOptions.lowBitAffine`
  (a lock-backed A/B switch the app sets at model load; one line in `supportsFusedDecode` reads it)
  and `Inkling2KernelSelfCheck`, which runs the fused kernels and the unfused `gatherQuantizedMM` path
  on a random layer per format and reports the differences. Upstream drops K2.

K1's reference and tests are in `ios/tools/kernels/`: a NumPy port of the kernels whose test fails
when the Metal source's SHA-256 changes, and `check_metal_kernels.py`, which compiles the generated
Metal in CI. Patches A/B are marked `[agent-console patch A/B]` in the source.

## Updating

Re-copy the four items above from a newer PR head (or the release that merges it), re-apply the
patches if still needed, and drop this directory in favour of a normal versioned package once
Inkling 2 ships in an mlx-swift-lm release.
