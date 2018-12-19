
Contributing 
============

OpenOA repository: http://github.com/NREL/OpenOA

## Coding Style

This code follows the PEP 8 style guide and uses ``pycodestyle`` to check for lint compliance.
The only exception is the line length limit of 120 characters. 

## Documentation Style

The canonical documentation for the PRUF analysis methodology is located in the
[sphinx](https://github.com/NREL/OpenOA/tree/release/v1/sphinx) directory.
Any changes to the analysis methodology should be discussed there or offline. Once a methodology change is decided,
create new tickets in this repository towards implementing the change.

## Repository Style

This repository is organized using the git-flow system. Branches are more-or-less organized as follows:

- master: Current release including all hot fixes.
- release/xxx: Development branches targeting a specific release. Tests should pass, but code may be unstable.
- feature/issue-xxx: Branch of active release branch, must reference a github issue number.
Features branches are not automatically tested and may contain broken code. Feel free to commit broken code to your own branch.

To work on a feature, please make a new feature branch of the active release branch.
Work out of that branch before submitting a pull request.
Complete pull requests should include both updated documentation and pass all unit tests and integration tests.

## Issue Tracking

New features, changes, enhancements, non-methodology features, and bug reports can be filed as new issues at any time.
Do your best to place the issue into a relevant milestone and project.
Care should be taken when submitting new issues close to a release date.
Any new issues in the current active project should be discussed at the next weekly standup at NREL.

## Pull Request

Pull requests must be made for any changes to be merged into release or master branches.

**Scope:** Encapsulate the changes of ideally one, or potentially a couple, issues. It is greatly preferable
to submit three small pull requests than it is to submit one large pull request. Write a complete description of these
changes in the pull request body.

**Tests:** Pass all tests as defined in Readme.md. Pull requests will be rejected if tests do not pass.

**Documentation:** Include any relevant changes to inline documentation.

**Coverage:** [Travis CI](https://travis-ci.org) will report the change in test coverage upon successful build.
Ensure this number is not negative.
