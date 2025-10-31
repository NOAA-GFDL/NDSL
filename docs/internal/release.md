# Release a new version

This internal documentation guides you through the process of releasing a new version of NDSL. It is very simple:

1. Click [create a release](https://github.com/NOAA-GFDL/NDSL/compare/main...develop?expand=1&template=release.md) and follow the steps in the release checklist.

## Patch release

Every now and then, we'll need to patch the currently released version of NDSL. To do so, follow these steps:

1. Create a branch from `main`.
2. Commit your changes on that branch.
3. Use the following URL <https://github.com/NOAA-GFDL/NDSL/compare/main...[your-branch-name]?expand=1&template=release-patch.md> and follow the steps in the patch release checklist.

As an example, you'd go and create branch `my-patches` from `main`

```bash
git checkout main
git switch -c my-patches
# do changes ...
git push
```

and in that case, the URL with the patch release template is: <https://github.com/NOAA-GFDL/NDSL/compare/main...my-patches?expand=1&template=release-patch.md>.
