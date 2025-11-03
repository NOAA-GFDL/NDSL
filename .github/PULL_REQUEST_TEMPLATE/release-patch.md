# Release NDSL version `YYYY.MM.PP`

This PR patches release `YYYY.MM.PP` because

1. reason
2. reason
3. ...

## Pre-release checklist

Things to do before the patch release. Helps to keep the fallout from this release as minimal as possible.

- [ ] setup a draft PR in [NOAA-GFDL/pace](https://github.com/NOAA-GFDL/pace) with updated submodules for `NDSL`, `pyFV3`, and `pySHiELD`.
  Don't merge yet - just let CI run and fix potential issues before the release. To be merged afterwards, see post-release checklist.

## Release checklist

What to do to actually release:

- [x] create this PR to merge changes from `my-patches` into `main`
  - use "squash merge"
- [ ] once merged, create a GitHub release and tag the new version
  - version format is `[year].[month].[patch]`. Increase the patch version, e.g. `2025.10.01` if this is patching the `2025.10.00` release.
  - let GitHub auto-generate release notes from the last tagged version
- [ ] send an announcement on Mattermost

## Post-release checklist

What to do after a release:

- [ ] update the pace PR from the pre-commit checklist to include the released version of NDSL and merge it.
- [ ] in NDSL, merge `main` back into `develop` (potentially adding a commit to fix the issue "properly")
