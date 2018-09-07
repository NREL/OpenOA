Instructions for Public Release
===============================

Create a cleaned version of the release branch:

```
git clone git@github.nrel.gov:Benchmarking/operational-analysis.git operational-analysis-public
git checkout release/beta
git branch release/beta-temp
git checkout release/beta-temp
```

Remove files we don't want in the public release:

```
rm contributing.md public_release.md
```

Create squashed version:

```
git commit -a -m "public release"
git checkout --orphan release/beta-public
```

Push to public repository:

```
git remote add public git@github.com:NREL/operational-analysis.git
git push --set-upstream public release/beta-public
```
