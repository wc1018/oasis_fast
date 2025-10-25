from dataclasses import dataclass
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional, Union

import numpy

__all__ = [
    'G_GRAVITY',
    'timer',
    'TimerContext',
    'get_min_unit_dtype',
    'ensure_dir_exists',
    'load_seeds',
]

# Gravitational constant
G_GRAVITY: float = 4.3e-09     # Mpc (km/s)^2 / M_sun


@dataclass(frozen=True)
class AnsiColor:
    """ANSI escape codes for colored and styled terminal output."""
    HEADER: str = "\033[35m"
    OKBLUE: str = "\033[34m"
    OKCYAN: str = "\033[36m"
    OKGREEN: str = "\033[32m"
    WARNING: str = "\033[33m"
    FAIL: str = "\033[31m"
    ENDC: str = "\033[0m"
    BOLD: str = "\033[1m"
    UNDERLINE: str = "\033[4m"
    BULLET: str = "\u25CF"


def timer(
    func: Optional[Callable] = None,
    *,
    fancy: bool = True,
    enabled: bool = True,
    precision: int = 3,
) -> Callable:
    """
    Decorator that measures and prints a function's execution time.

    Parameters
    ----------
    func : Callable, optional
        Function to be timed.
    fancy : bool, optional
        Whether to use colored output, by default True.
    enabled : bool, optional
        Whether timing is enabled, by default True.
    precision : int, optional
        Decimal precision for time display, by default 3.

    Returns
    -------
    Callable
        Decorated function or decorator factory.

    Examples
    --------
    >>> @timer
    ... def slow_function():
    ...     time.sleep(1)

    >>> @timer(fancy=False, precision=2)
    ... def another_function():
    ...     return 42
    """
    if func is None:
        return partial(timer, fancy=fancy, enabled=enabled, precision=precision)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not enabled:
            return func(*args, **kwargs)

        fmt = '%Y-%m-%d %H:%M:%S'
        start_time = datetime.now()
        perf_start = perf_counter()

        try:
            result = func(*args, **kwargs)
        finally:
            finish_time = datetime.now()
            elapsed = perf_counter() - perf_start

            hours = int(elapsed // 3600)
            remaining_seconds = elapsed % 3600
            minutes = int(remaining_seconds // 60)
            seconds = remaining_seconds % 60
            miliseconds = int(10**precision * (seconds - int(seconds)))
            elapsed_str = f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{miliseconds:03d}"

            if fancy:
                print(
                    f"\t{AnsiColor.BOLD}Process:{AnsiColor.ENDC} "
                    f"{AnsiColor.FAIL}{func.__name__}{AnsiColor.ENDC}\n"
                    f"Start:  {AnsiColor.OKBLUE}{start_time.strftime(fmt)}{AnsiColor.ENDC}\n"
                    f"Finish: {AnsiColor.OKBLUE}{finish_time.strftime(fmt)}{AnsiColor.ENDC}\n"
                    f"{AnsiColor.BULLET}{AnsiColor.BOLD}{AnsiColor.OKGREEN} Elapsed: "
                    f"{AnsiColor.ENDC}{AnsiColor.WARNING}{elapsed_str}{AnsiColor.ENDC}"
                )
            else:
                print(
                    f"\tProcess: {func.__name__}\n"
                    f"Start:  {start_time.strftime(fmt)}\n"
                    f"Finish: {finish_time.strftime(fmt)}\n"
                    f"{AnsiColor.BULLET} Elapsed: {elapsed_str}"
                )

        return result

    return wrapper


class TimerContext:
    """
    Context manager for timing code blocks.

    Examples
    --------
    >>> with TimerContext("data processing"):
    ...     # some time-consuming operation
    ...     time.sleep(1)
    """

    def __init__(self, name: str = "operation", fancy: bool = True, precision: int = 3):
        self.name = name
        self.fancy = fancy
        self.precision = precision
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = perf_counter() - self.start_time
        elapsed_str = f"{self.elapsed:.{self.precision}f}s"

        if self.fancy:
            print(
                f"{AnsiColor.BULLET}{AnsiColor.BOLD}{AnsiColor.OKGREEN}"
                f"{self.name} completed in {AnsiColor.ENDC}"
                f"{AnsiColor.WARNING}{elapsed_str}{AnsiColor.ENDC}"
            )
        else:
            print(f"{self.name} completed in {elapsed_str}")


def get_min_unit_dtype(num: int) -> numpy.dtype:
    """
    Determine the minimum unsigned integer dtype to represent a number.

    Parameters
    ----------
    num : int
        Non-negative integer value.

    Returns
    -------
    numpy.dtype
        Minimum unsigned integer dtype that can represent the number.

    Raises
    ------
    TypeError
        If input is not a non-negative integer.
    OverflowError
        If number exceeds uint64 maximum value.

    Examples
    --------
    >>> get_min_uint_dtype(255)
    dtype('uint8')
    >>> get_min_uint_dtype(70000)
    dtype('uint32')
    """
    if not isinstance(num, (int, numpy.number)) or num < 0:
        raise TypeError("Input must be a non-negative integer")

    # Check in order of size
    for dtype in [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]:
        if num <= numpy.iinfo(dtype).max:
            return dtype

    raise OverflowError(f"Number {num} exceeds maximum uint64 value")


def ensure_dir_exists(
    path: Union[str, Path],
    verbose: bool = False,
    parents: bool = True,
) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str or Path
        Directory path to ensure exists.
    verbose : bool, optional
        Whether to print status messages, by default False.
    parents : bool, optional
        Whether to create parent directories, by default True.

    Returns
    -------
    Path
        Absolute path to the directory.

    Raises
    ------
    PermissionError
        If lacking permissions to create directory.
    OSError
        If directory creation fails for other reasons.

    Examples
    --------
    >>> ensure_dir_exists("./data/output")
    PosixPath('/absolute/path/to/data/output')
    """
    path_obj = Path(path).resolve()

    if path_obj.exists():
        if not path_obj.is_dir():
            raise OSError(f"Path exists but is not a directory: {path_obj}")
        if verbose:
            print(f"Directory already exists: {path_obj}")
        return path_obj

    try:
        path_obj.mkdir(parents=parents, exist_ok=True)
        if verbose:
            print(f"Created directory: {path_obj}")
        return path_obj
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied creating directory {path_obj}") from e
    except OSError as e:
        raise OSError(f"Failed to create directory {path_obj}: {e}") from e


def _validate_inputs_positive_number(value, name):
    """Validate that a value is a positive number."""
    if not isinstance(value, (int, float, numpy.number)):
        raise TypeError(f"{name} must be numeric")
    if value < 0:
        raise ValueError(f"{name} must be zero or positive")


def _validate_inputs_positive_number_non_zero(value, name):
    """Validate that a value is a positive number."""
    _validate_inputs_positive_number(value, name)
    if value == 0 or value == 0.:
        raise ValueError(f"{name} must be non-zero")


def _validate_inputs_coordinate_arrays(arr, name):
    """Validate coordinate arrays and raise appropriate errors."""
    if not isinstance(arr, numpy.ndarray):
        raise TypeError("Input must be a numpy array")

    if arr.shape[0] == 0:
        raise ValueError(f"Input {name} must contain at least one particle")

    if (arr.ndim == 1 and arr.shape[0] != 3) or (arr.ndim == 2 and arr.shape[1] != 3) or \
            len(arr.shape) > 2:
        raise ValueError(
            f"Input {name} must have shape (3,) or (n_particles, 3)")


def _validate_inputs_existing_path(path):
    """Validate that a path exists and is a directory."""
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or Path object")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")


def _validate_inputs_boxsize_minisize(boxsize, minisize):
    """Validate boxsize and minize and raise appropriate errors."""
    # Validate both are positive numbers
    _validate_inputs_positive_number_non_zero(boxsize, "boxsize")
    _validate_inputs_positive_number_non_zero(minisize, "minisize")

    # Check that minisize is not larger than boxsize
    if minisize > boxsize:
        raise ValueError("minisize cannot be larger than boxsize")


def _validate_inputs_mini_box_id(mini_box_id, cells_per_side):
    """Validate mini-box ID and raise appropriate errors."""

    # Check mini_box_id is an integer
    if not isinstance(mini_box_id, (int, numpy.integer)):
        raise TypeError("mini_box_id must be an integer")

    # Check mini_box_id is a positive number
    _validate_inputs_positive_number(mini_box_id, 'mini_box_id')

    # Check cells_per_side is a positive integer
    if not isinstance(cells_per_side, (int, numpy.integer)):
        raise TypeError("cells_per_side must be a positive integer")

    if cells_per_side is None or cells_per_side <= 0:
        raise ValueError("cells_per_side must be a positive integer")

    # Validate mini_box_id is within valid range
    total_mini_boxes = cells_per_side**3
    if mini_box_id >= total_mini_boxes:
        raise ValueError(
            f"mini_box_id {mini_box_id} exceeds maximum valid ID {total_mini_boxes - 1} "
            f"for grid with {cells_per_side}³ = {total_mini_boxes} mini-boxes"
        )


def _validate_inputs_box_partitioning(positions, velocities, uid, props):
    """Validate function inputs and raise appropriate errors."""
    # Check array shapes
    _validate_inputs_coordinate_arrays(positions, "positions")
    _validate_inputs_coordinate_arrays(velocities, "velocities")

    if uid.ndim != 1:
        raise ValueError("uid must be a 1D array")

    n_particles = positions.shape[0]
    if velocities.shape[0] != n_particles or uid.shape[0] != n_particles:
        raise ValueError(
            "positions, velocities, and uid must have the same length")

    # Validate props structure if provided
    if props is not None:
        if not isinstance(props, tuple) or len(props) != 3:
            raise ValueError(
                "props must be a tuple of (arrays, labels, dtypes)")

        arrays, labels, dtypes = props
        if not (isinstance(arrays, (list, tuple)) and
                isinstance(labels, (list, tuple)) and
                isinstance(dtypes, (list, tuple))):
            raise ValueError("props must contain three lists")

        if not (len(arrays) == len(labels) == len(dtypes)):
            raise ValueError("All lists in props must have the same length")

        for i, arr in enumerate(arrays):
            if not isinstance(arr, numpy.ndarray):
                raise ValueError(f"props array {i} must be a numpy array")
            if arr.shape[0] != n_particles:
                raise ValueError(
                    f"props array {i} must have {n_particles} elements")


def _validate_inputs_load(
    mini_box_id: int,
    boxsize: float,
    minisize: float,
    load_path: str,
    padding: float,
) -> None:
    """Validate inputs for load functions."""
    # Validate mini_box_id
    _validate_inputs_mini_box_id(
        mini_box_id, int(numpy.ceil(boxsize / minisize)))

    # Validate boxsize and minisize
    _validate_inputs_boxsize_minisize(boxsize, minisize)

    # Validate load_path
    _validate_inputs_existing_path(load_path)

    # Validate padding
    _validate_inputs_positive_number_non_zero(padding, "padding")


def _validate_inputs_seed_data(seed_data):
    """Validate inputs for functions taking seed data."""
    if len(seed_data) != 4:
        raise ValueError("seed_data must be a tuple of 4 elements")

    position, velocity, mass, radius = seed_data

    _validate_inputs_coordinate_arrays(position, "seed positions")
    _validate_inputs_coordinate_arrays(velocity, "seed velocities")

    # Check all inputs have the same number of elements
    n_items = len(position)
    if n_items == 1:
        mass = numpy.asarray(mass)
        radius = numpy.asarray(radius)

    if not (mass.size == radius.size == len(position) == len(velocity)):
        raise ValueError("All elements in seed_data must have the same length")



###
