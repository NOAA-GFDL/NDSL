# Release a new version

This internal documentation guides you through the process of releasing a new version of NDSL. It is very simple:

1. Click [create a release](https://github.com/NOAA-GFDL/NDSL/compare/main...develop?expand=1&template=release.md) and follow the steps in the release checklist.

2. After merging that PR, create a GitHub release and tag the new version
    - version format is `[year].[month].[patch]`, e.g. `2025.10.00`
  - let GitHub auto-generate release notes from the last tagged version

3. Send an announcement on Mattermost
