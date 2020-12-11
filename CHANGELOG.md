# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
the entire project sticks to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1]

### Added
- More tests
- Coverage reports
- flake8 linting checks
- Docstring tests

### Changed
- Some docstring examples
- Fixed typos in the docstrings


## [0.1.0]

### Added
- Flat module for the predefined relevance judgements
- A simple example

### Removed
- Removed the separate ranking module

### Changed
- The structure of tests
- The default value of `k` is now `None`
- Moved `pandas` to optional dependencies

## [0.0.6]

### Added
- Precision and ndcg scores
- Small fixes to the docstrings
- Add coverage and IoU scores
- Add Average Precision score
- Add support for custom relevance functions
- The reciprocal rank and recall metrics

### Changed
- Move readme to `.rst`
- Simplify the `topk` tests

## [0.0.5]

### Added
- The reciprocal rank and recall metrics

## [0.0.4]

### Added
- The metainformation about the package

## [0.0.3]

### Changed
- Removed the test server from the deployment

## [0.0.2]

### Changed
- Moved the `ndcg` and `wdcg` scores to `ranking` module.

## [0.0.1]

### Added
- Support for automatic deployment to PyPI

### Changed
- Removed python information
- Package naming
