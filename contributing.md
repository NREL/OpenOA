
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

- release/xxx: Development branches targeting a specific release. Tests should pass, but code may be unstable.
- feature/issue-xxx: Branch of active release branch, must reference a github issue number.
Features branches are not automatically tested and may contain broken code. Feel free to commit broken code to your own branch.

Please note that our public repository does not have a master or a develop branch. Those branches are hosted on a
separate, private repository for the NREL team.

To work on a feature, please make a new feature branch based on the target release branch. If you're working externally
to NREL, please fork OpenOA first and then create a feature branch in your own copy of the repository.
Work out of the feature branch before submitting a pull request. Be sure to periodically merge the target release
branch into your feature branch to avoid conflicts in the pull request.

When the feature branch is ready, make a pull request through the Github.com UI.

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

OpenOA uses pytest and the built in unittest framework. To run tests, navigate to the OpenOA directory and run:

```
python setup.py test
```

