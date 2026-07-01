# Merge breaking changes in NDSL and related repositories

Disclaimer: Whenever possible, we try to avoid breaking changes. Sometimes, however, they will be necessary or very much inconvenient such that we'll make an exception to this general rule.

Breaking changes are incompatible with our standard procedure (i.e. current CI setup) because we have tests running in both directions of the dependency-chain: NDSL runs some `pyFV3`, `pySHiELD` and `pace` tests as part of the CI. Those tests install the latest release version of NDSL for its testing. This dependency on the latest release can be broken (and later restored) as described in this document.

## Phase 1

- [ ] PR #1: Create a branch on `NDSL` to bring in the breaking changes.
- [ ] PR #2: Create a branch on `pyFV3` that fixes the breaking changes.
- [ ] PR #3: Create a branch on `pySHiELD` that fixes the breaking changes.
- [ ] PR #4: Create a branch on `pace` that fixes the breaking changes.

## Phase 2

- [ ] In PR #1, change the targets of `.github/workflows/fv3_translate_tests.yaml`, `.github/workflows/pace_tests.yaml` and `.github/workflows/pace_tests.yaml` to point to the branches created above, e.g.

```yaml
jobs:
  fv3_translate_tests:
    uses: NOAA-GFDL/pyFV3/.github/workflows/translate.yaml@develop
```

becomes

```yaml
jobs:
  fv3_translate_tests:
    uses: twicki/pyFV3/.github/workflows/translate.yaml@your_breaking_change
```

- [ ] In PR #2, change the targets of `.github/workflows/pace_tests.yaml` and `.github/workflows/pyshield_tests.yaml` to point to the branches created above.
- [ ] In PR #3, change the targets of `.github/workflows/pace_tests.yaml` and the pySHiELD target of `.github/workflows/translate.yaml` o point to the branches created above.

## Phase 3

With these changes, all PRs 1-3 should be passing CI now and can be merged.

- [ ] Merge PR #1
- [ ] Merge PR #2
- [ ] Merge PR #3

## Phase 4

- [ ] In PR #4, update the submodules in `pace` to point to the updated HEADs of `NDSL`, `pyFV3` and `pySHiELD`.

## Phase 5

- [ ] Merge PR #4.

## Phase 6

With this, all the functionality has been merged and propagated everywhere, so a reset to all `develop`-branches is possible:

- [ ] PR #5: create a PR in `NDSL` switching `.github/workflows/fv3_translate_tests.yaml`, `.github/workflows/pace_tests.yaml` as well as `.github/workflows/shield_tests.yaml` back to `NOAA-GFDL[...]@develop`, reverting phase 2.
- [ ] PR #6: create a PR in `pyFV3` switching `.github/workflows/pace_tests.yaml` and `.github/workflows/pyshield_tests.yaml` back to `NOAA-GFDL[...]@develop`, reverting phase 2.
- [ ] PR #7: create a PR in `pySHiELD` switching `.github/workflows/pace_tests.yaml` and `.github/workflows/translate.yaml` back to `NOAA-GFDL[...]@develop`, reverting phase 2.

## Phase 7

- [ ] PR #8: Create a PR in pace updating all the submodules to be on the develop branches again instead of the branches created in Phase 4.
