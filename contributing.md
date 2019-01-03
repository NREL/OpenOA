
Contributing 
============

## Issue Tracking

New features, changes, enhancements, non-methodology features, and bug reports can be filed as new issues in the
github issue tracker at any time.

## Repository

OpenOA repository: http://github.com/NREL/OpenOA

This repository is organized using the git-flow system. Branches are more-or-less organized as follows:

- master: Current release including all hot fixes.
- release/xxx: Development branches targeting a specific release. Tests should pass, but code may be unstable.
- feature/issue-xxx: Branch of active release branch, must reference a github issue number.
Features branches are not automatically tested and may contain broken code. Feel free to commit broken code to your own branch.

To work on a feature, please make a new feature branch of the active release branch. If you're working externally
to NREL, please fork OpenOA first and then create a feature branch in your own copy of the repository.
Work out of that branch before submitting a pull request.
Complete pull requests should include both updated documentation and pass all unit tests and integration tests.

## Pull Request

Pull requests must be made for any changes to be merged into release or master branches.

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