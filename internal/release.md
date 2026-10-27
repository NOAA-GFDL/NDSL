# Release a new version

This internal documentation guides you through the process of releasing a new version of NDSL. It is very simple:

1. Go to the [releases page](https://github.com/NOAA-GFDL/NDSL/releases).
2. Click _Draft a new release_ (located top right at the time of writing).
    - Tag: Create a new one. We use the following convention `YYYY.MM.PP` where
        - `YYYY` is a 4-digit year, e.g. `2026`;
        - `MM` is a 2-digit month, e.g. `03`
        - `PP` denotes the patch level, mostly just `00`
    - Target: `develop`
    - Release title: `NDSL YYYY.MM.PP`
3. Let GitHub auto-generate the release notes from the last tagged version.
4. Add `Highlights` and `Deprecations` / `Breaking changes` sections as needed.
5. Select as "latest release" and publish the release.
6. Follow the (after) release checklist below.

## Release checklist

Below is a short (if it get too long, consider automating parts) release checklist for whoever does the release

- [ ] Create a release as above.
- [ ] Send the announcement on Mattermost.
- [ ] Update `pyFV3`, `pySHiELD`, and `pace` (in that order) with the newly released version of NDSL.
- [ ] in NDSL, merge breaking changes of deprecated features/functions (search for e.g. `DeprecationWarning`).
- [ ] in the NDSL docs, update the latest release and check if the roadmap needs updating

## Patch release

Every now and then, we'll need to patch the currently released version of NDSL. To do so, follow the steps below.

1. Create a branch from the tag you want to patch.
2. Commit your changes on that branch.
3. Test your changes, e.g. by creating a draft PR with your changes against `develop` such that CI can run.
4. Create a release (see above) from that branch (the branch can be deleted once the tag is made).
5. Follow the release checklist above.
  - Note: since the patch changes aren't merged into `develop` yet, auto-generating release notes wont' work.
6. Merge the PR from step 3 to bring the fix to `develop` or make a better fix in `develop`.
