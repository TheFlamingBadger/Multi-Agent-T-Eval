import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="builtin type swigvarlink has no __module__ attribute",
)
