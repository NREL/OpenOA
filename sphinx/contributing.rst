.. _contributing:


Contributing
************

OpenOA repository: http://github.com/NREL/OpenOA


Repository Layout
=================

This repository is organized using the git-flow system. Branches are more-or-less organized as follows:

- master: Current stable release including any hotfixes. Documentation is up to date.
- release/xxx: Development branches targeting a specific release. Tests should pass. Branch may be unstable and out of sync with documentation.
- feature/issue-xxx: Branch of active release branch. Should reference a github issue number. Features branches are not automatically tested and may contain broken code.

To contribute some code, please make a new feature branch from the target release branch.
When done, submit a pull request into the target release branch.
Pull requests should include updated documentation and pass all unit tests.

Pull Requests
=============

Pull requests must be made for any changes to be merged into release or master branches.

**Scope:** Encapsulate the changes of ideally one, or potentially a couple, issues. It is greatly preferable
to submit three small pull requests than it is to submit one large pull request. Write a complete description of these
changes in the pull request body.

**Tests:** Pass all tests as defined in Readme.md. Pull requests will be rejected if tests do not pass.

**Documentation:** Include any relevant changes to inline documentation.

