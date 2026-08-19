# Release NDSL version `YYYY.MM.00`

## Pre-release checklist

Things to do before the release. Helps to keep the fallout from this release as minimal as possible.

- [ ] setup a draft PR in [NOAA-GFDL/pace](https://github.com/NOAA-GFDL/pace) with updated submodules for `NDSL`, `pyFV3`, and `pySHiELD`.
  Don't merge yet - just let CI run and fix potential issues before the release. To be merged afterwards, see post-release checklist.

## Release checklist

- [ ] Merge this PR
- [ ] Create a GitHub release and tag the new version
  - version format is `[year].[month].[patch]`, e.g. `2025.10.00`
  - let GitHub auto-generate release notes from the last tagged version

## Post-release checklist

What to do after a release:

- [ ] update the pinned version of [pyFV3](https://github.com/NOAA-GFDL/PyFV3/) to the new release-tag
- [ ] update the pinned version of [pySHiELD](https://github.com/NOAA-GFDL/pySHiELD) to the new release-tag
- [ ] update the pace PR from the pre-commit checklist to include the released version of NDSL and update the submodules before merging it.
- [ ] merge breaking changes in NDSL (e.g. search for deprecation warnings)
