# Release NDSL version `2025.XX.00`

## Pre-release checklist

Things to do before the release. Helps to keep the fallout from a release as minimal as possible.

- [ ] setup a draft PR in [NOAA-GFDL/pace](https://github.com/NOAA-GFDL/pace) with updated submodules for `NDSL`, `pyFV3`, and `pySHiELD`.
  Don't merge yet - just let CI run and fix potential issues before the release. To be merged afterwards, see post-release checklist.

## Release checklist

What to do to actually release:

- [x] create this PR to merge changes from `develop` into `main`
  - merge as "Merge commit"
- [ ] once merged, create a GitHub release and tag the new version
  - version format is `[year].[month].[patch]`, e.g. `2025.10.00`
  - let GitHub auto-generate release notes from the last tagged version
- [ ] send an announcement on Mattermost

## Post-release checklist

What to do after a release:

- [ ] update the pace PR from the pre-commit checklist to include the released version of NDSL and merge it.
- [ ] merge breaking changes in NDSL (e.g. search for deprecation warnings)
