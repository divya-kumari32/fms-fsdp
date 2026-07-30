# Revert notes — branch `ra-ac-grouping`

Goal: add optional **grouped activation checkpointing** to RA so N=8 fits in 80GB
(N=8 OOMs by ~4.6 GiB on the current per-block AC). Feature is **off by default**
(`ra_ac_group_size=0`) and the off-path is byte-equivalent to the old code
(verified: CPU forward + grad diff = 0.0 between group_size=0 and group_size=2).

## To revert completely
```bash
git checkout ra-experiments      # the pre-refactor branch (HEAD 498dd82)
# or, to drop the branch:
git branch -D ra-ac-grouping
```
Nothing on `ra-experiments` is touched. The N=4 full-run YAML does NOT set
`ra_ac_group_size`, so it defaults to 0 → identical behavior even if run from this branch.

## Exact changes (3 files)

1. **`fms_fsdp/config/training.py`** — added field after `ra_num_slots`:
   ```python
   ra_ac_group_size: int = 0   # 0 = disabled (use standard per-block AC handler)
   ```

2. **`fms_fsdp/models/ra_llama.py`**
   - added `from torch.utils.checkpoint import checkpoint`
   - `__init__` gained `ac_group_size=0` kwarg, stored as `self.ac_group_size`
   - extracted the old per-layer loop body into `_run_layers(self, cache, position_ids, start, end)`
     → returns `(x, cache)`. **Loop body is byte-identical** to the old forward
     (read `_depth_attend` → block → accumulate write `slots[i%N] += contribution`).
   - `forward`: if `self.ac_group_size > 0 and self.training`, iterate layers in
     regions of `ac_group_size` calling `checkpoint(self._run_layers, ..., use_reentrant=False)`;
     otherwise call `_run_layers` once over all layers (the old behavior).

3. **`main_training_ra_llama.py`**
   - both `RALLaMA(...)` instantiations now pass `ac_group_size=cfg.ra_ac_group_size`
   - AC block: when `cfg.ra_ac_group_size > 0`, skip `apply_selective_ac` (model
     self-checkpoints in groups → avoid double-checkpointing); else unchanged.

## How to enable (N=8 smoke / full run)
Add to `mainProgram`:  `--ra_ac_group_size=2`  (with `--fsdp_activation_checkpointing=True`).

## Also on this branch
- `REVERT_ra_ac_grouping.md` (this file) — delete on revert if desired.
