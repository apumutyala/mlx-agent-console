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

Both code patches are marked `[agent-console patch A/B]` in the source.

## Updating

Re-copy the four items above from a newer PR head (or the release that merges it), re-apply the
patches if still needed, and drop this directory in favour of a normal versioned package once
Inkling 2 ships in an mlx-swift-lm release.
