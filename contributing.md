
Contributing
============

## Issue Tracking

New feature requests, changes, enhancements, non-methodology features, and bug reports can be filed as new issues in the
[Github.com issue tracker](https://github.com/NREL/OpenOA/issues) at any time. Please be sure to fully describe the
issue.

For other issues, please email the OpenOA distribution list at `openoa@nrel.gov`.

## Repository

The OpenOA repository is hosted on Github, and located here: http://github.com/NREL/OpenOA

This repository is organized using a modified git-flow system. Branches are organized as follows:

- master: Stable release version. Must have good test coverage and may not have all the newest features.
- develop: Development branch which contains the newest features. Tests must pass, but code may be unstable.
- feature/xxx: Branch from develop, should reference a github issue number.

To work on a feature, please fork OpenOA first and then create a feature branch in your own fork.
Work out of this feature branch before submitting a pull request.
Be sure to periodically synchronize the upstream develop branch into your feature branch to avoid conflicts in the pull request.

When the feature branch is ready, make a pull request to NREL/OpenOA through the Github.com UI.

## Pull Request

Pull requests must be made for any changes to be merged into release branches.
They must include updated documentation and pass all unit tests and integration tests.
In addition, code coverage should not be negatively affected by the pull request.

**Scope:** Encapsulate the changes of ideally one, or potentially a couple, issues. It is greatly preferable
to submit three small pull requests than it is to submit one large pull request. Write a complete description of these
changes in the pull request body.

**Tests:** Must pass all tests. Pull requests will be rejected if tests do not pass.

**Documentation:** Include any relevant changes to inline documentation, as well as any changes to the RST files
located in /sphinx.

**Coverage:** The testing framework (described below) will generate a coverage report. Please ensure that your
work is fully covered.

**Changelog:** For pull requests that encapsulate a user-facing feature, or is significant to users of OpenOA for some other reason, please add a line to CHANGELOG.md in the [Unreleased] section.

## Coding Style

This code follows the PEP 8 style guide and uses the ``pycodestyle`` linter to check for compliance.
The only exception is the line length limit of 120 characters.

```
pylint --max-line-length=120 operational_analysis
```

## Documentation Style

Documentation is written using RST, and is located both inline and within the /sphinx directory.
Any changes to the analysis methodology should be discussed there or offline. Once a methodology change is decided,
create new tickets in this repository towards implementing the change.

## Testing

All code should be paired with a corresponding unit or integration test.
OpenOA uses pytest and the built in unittest framework.
For instructions on running tests, please see the [Readme](https://github.com/NREL/OpenOA/tree/develop#Testing).
