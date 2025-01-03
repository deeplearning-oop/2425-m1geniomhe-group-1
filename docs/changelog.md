# Changelog

## [1.0.0] - 2024-12-28
- First logged version of the library ✅
- Tensor class in Tensor module created with: full Tensor functionalities, error handling and notebook tests + pytorch ui comparisons🚀
- dtype Enum class in Tensor module created, involving 2 datatypes for now: int64 and float64
- Primary documentation of the steps added in tests/ 🧪
- Primary repository structure design to include: requirements.txt, LICENCE.md, VERSION and docs/


## [1.0.1] - 2024-12-30
- Allowed numpy ndarray input to tensor through the static method validating the data input
- Allowed direct conversion between tensor and numpy

## [1.0.2] - 2025-01-03
- Added abstract class Dataset
- Added MNIST child of Dataset:
- 	Performing webscraping data, extraction raw -> IDX -> ndarray -> Tensor
- 	Implemented abstract methods with capability to viz every item

