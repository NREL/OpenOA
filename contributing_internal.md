
Contributing (Internal Version)
===============================

## Change Request Process

### Methodology Changes

The canonical documentation for the pruf analysis methodology is located in the
[Operational Analysis Report](https://github.nrel.gov/Benchmarking/operational-analysis-report) repository.
Any changes to the analysis methodology should be discussed there or offline. Once a methodology change is decided,
create new tickets in this repository towards implementing the change.

### Bug Reports

Submit bugs as new issues assigned to the current project.

### Enhancements

Submit enhancements / non-methodology feature requests as new issues assigned to any future project.

### Iterations

Every Thursday we will have a standup meeting to discuss new tickets, assess progress on active tickets, and assess our
progress towards the release date.

## Git Repository

### Branch Layout

This repository is organized using the git-flow system. Branches are more-or-less organized as follows:

- master: Current release including all hotfixes.
- release/xxx: Development branches targeting a specific release. Tests should pass, but code may be unstable.
- feature/issue-xxx: Branch of active release branch, must reference a github issue number. Feel free to commit broken code to your own branch.

To work on a feature, please make a branch of the active release branch. Work out of that branch before submitting a pull request.
Complete pull requests should include both updated documentation and pass all unit tests.

### Pull Requests

Pull requests must be made for any changes to be merged into release or master branches.

**Scope:** Encapsulate the changes of ideally one, or potentially a couple, issues. It is greatly preferable
to submit three small pull requests than it is to submit one large pull request. Write a complete description of these
changes in the pull request body.

**Tests:** Pass all tests as defined in Readme.md. Pull requests will be rejected if tests do not pass.

**Documentation:** Include any relevant changes to inline documentation.

**Coverage:** Jenkins will report the change in test coverage upon successful build. Ensure this number is not negative.

### Issues

New features, changes, and bug reports can be filed as new issues at any time. Do your best to place the issue into a
relevant milestone and project. Care should be taken when submitting new issues close to a release date.
Any new issues in the current active project should be discussed at the next standup.

### Projects

The project board contains all of the Issues we're working towards in a given Milestone Iteration.
Ideally, we only update project boards during weekly standups.

### Milestones

Deadlines for releases. Milestones represent large goals for feature sets / use cases of the software.
We discuss milestone goals at weekly standups and can change them at PRUF internal meetings.
