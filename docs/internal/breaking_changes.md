# Merge breaking changes in the DSL-stack

When it is required to make breaking changes, the standard procedure does not work because we have tests running in both directions of the dependency-chain. So NDSL does run some pyFV3, pace and pySHiELD tests, pyFV3 installs NDSL for its testing.
In order to solve this, the breaking changes coming from the DSL side (whether directly from NDSL or from an update to GT4Py going into NDSL) - should come with a change to point to branches of pace, shield and pyFV3:


### Phase 1
- [ ] PR #1: Create a branch on NDSL to bring in the breaking changes
- [ ] PR #2: Create a branch on pyFV3 that fixes the breaking changes
- [ ] PR #3: Create a branch on pySHiELD that fixes the breaking changes
- [ ] PR #4: Create a branch on pace that fixes the breaking changes
---
### Phase 2
- [ ] in PR #1: Change the targets of `.github/workflows/fv3_translate_tests.yaml`, `.github/workflows/pace_tests.yaml` and `.github/workflows/pace_tests.yaml` to point to the branches created above:
```
jobs:
  fv3_translate_tests:
    uses: NOAA-GFDL/pyFV3/.github/workflows/translate.yaml@develop
```
becomes:
```
jobs:
  fv3_translate_tests:
    uses: twicki/pyFV3/.github/workflows/translate.yaml@your_breaking_change
```
- [ ] in PR #2: Change the targets of `.github/workflows/pace_tests.yaml` and `.github/workflows/pyshield_tests.yaml` to point to the branches created above
- [ ] in PR #3: Change the targets of `.github/workflows/pace_tests.yaml` and the pySHiELD target of `.github/workflows/translate.yaml` o point to the branches created above
---
### Phase 3
With these changes, all PR's 1-3 should be passing and can be merged.
- [ ] Merge PR #1
- [ ] Merge PR #2
- [ ] Merge PR #3
---
### Phase 4
- [ ] in PR #4: update the submodules in pace to point to the new HEAD's of NDSL, pyFV3 and pySHiELD
---
### Phase 5
- [ ] merge PR #4
---
### Phase 6
With this, all the functionality has been merged and propagated everywhere, so a reset to all develop-branches is possible:
- [ ] PR #5: create a PR in NDSL switching `.github/workflows/fv3_translate_tests.yaml`, `.github/workflows/pace_tests.yaml` as well as  `.github/workflows/shield_tests.yaml` back to `NOAA-GFDL[...]@develop`, reverting phase 2
- [ ] PR #6: create a PR in pyFV3 switching `.github/workflows/pace_tests.yaml` and `.github/workflows/pyshield_tests.yaml` back to `NOAA-GFDL[...]@develop`, reverting phase 2
- [ ] PR #7: create a PR in pySHiELD switching `.github/workflows/pace_tests.yaml` and `.github/workflows/translate.yaml` back to `NOAA-GFDL[...]@develop`, reverting phase 2
---
### Phase 7
- [ ] PR #8: Create a PR in pace updating all the submodules to be on the develop branches again instead of the branches created in Phase 4