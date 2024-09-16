# Contributing to nleis.py
🔋 Welcome to the Open-Source nonlinear electrochemical impedance analysis Community! 🎉

We are excited to have you contribute to the nleis.py project! Contributions of all kinds are welcome, including bug fixes, feature requests, code reviews, new features, examples, documentation improvements, and community engagement.

This document outlines how to contribute and the development workflow for the project.

## Bug Reports and Suggestions
If you find a bug in the code or a mistake in the [documentation](https://nleispy.readthedocs.io/en/latest/index.html) or want a new feature, you can help us by creating an [issue](https://github.com/yuefan98/nleis.py/issues) in our repository, or even submit a [pull request](https://github.com/yuefan98/nleis.py/pulls).

# Development Guide

nleis.py is developed in conjunction with [impedance.py](https://github.com/ECSHackWeek/impedance.py/tree/main), with the goal of expanding its capabilities to nonlinear impedance analysis. When contributing to nleis.py, please also refer to and follow the [impedance.py contribution guidelines](https://github.com/ECSHackWeek/impedance.py/blob/main/CONTRIBUTING.md).

## Repository Setup

1.  To work on the nleis.py package, you should first fork the repository on GitHub using the button on the top right of the yuefan98/nleis.py repository.

2.  You can then clone the fork to your computer

```bash
git clone https://github.com/<YourGitHubUsername>/nleis.py
```

3.  Make your changes and commit them to your fork (for an introduction to git, checkout the [tutorial from the ECS Hack Week](https://github.com/ECSHackWeek/ECSHackWeek_Dallas/blob/master/Version_Control.pptx))

For example,
```bash
git add changedfiles
git commit
git push
```

4.  [Submit a Pull Request](https://github.com/yuefan98/nleis.py/pulls) (make sure to write a good message so the reviewer can understand what you're adding!) via GitHub.

5.  Add yourself to the list of collaborators (you can use the [all-contributors bot](https://allcontributors.org/docs/en/bot/usage))! You rock!

## Continuous Integration

nleis.py uses GitHub Actions for Continuous Integration (CI) testing. Every time you submit a pull request, a series of tests will be automatically run to ensure that the changes do not introduce any bugs :bug:. Your PR will not be merged until it passes all tests. While you can certainly wait for the results of these tests after submitting a PR, you can run these tests locally to speed up the process.

### Code Style - PEP 8

We use flake8 to enforce [PEP 8](https://www.python.org/dev/peps/pep-0008/) conformance, which is a style guide for Python code. To ensure your code follows PEP 8 guidelines, you can install and run flake8 locally:

```
conda install flake8
cd nleis.py/
flake8
```
:warning: if there is any output here, fix the errors and try running flake8 again.

### Unit Tests

We aim for high code coverage and thorough testing. Before submitting a pull request, please make sure your changes pass all existing tests, and add new tests for any new functionality you introduce. You can run tests locally using pytest:

```
conda install pytest
cd nleis.py/
pytest
```
:warning: you should see all tests pass, if not try fixing the error or file an issue.
