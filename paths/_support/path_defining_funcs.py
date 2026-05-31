###############################################################################
## CARTO PATHS
###############################################################################
from pathlib import Path
from typing import Optional, Union, List

def carto_paths(
    base_dir: Union[str, Path],
    filename: Optional[str] = None,
) -> List[Path]:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        return []

    return [base_dir / filename]



###############################################################################
## RF PATHS
###############################################################################
from pathlib import Path
from typing import Optional, Union, List

def RF_paths(
    base_dir: Union[str, Path],
    band_selection: str,
    hab_selection: str,
    filename: Optional[Union[str, Path]] = None,
) -> List[Path]:
    base_dir = Path(base_dir)
    out_dir = base_dir / band_selection / hab_selection
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        return [out_dir]

    return [out_dir / Path(filename)]



###############################################################################
## RF TIMEFRAME PATHS
###############################################################################
from pathlib import Path
from typing import Union, Optional, List

def RF_paths_quarters(
    base_dir: Union[str, Path],
    band_selection: str,
    hab_selection: str,
    train_split_attempt: str,
    timeframes: List[str],
) -> List[Path]:
    """
    Returns 4 Paths (Q1..Q4 by default) for:
    stack_b2348__{hab_selection}_{train_split_attempt}_{timeframe}__rstr.tif
    """
    paths: List[Path] = []
    for timeframe in timeframes:
        paths.extend(
            RF_paths(
                base_dir=base_dir,
                band_selection=band_selection,
                hab_selection=hab_selection,
                filename=f"stack_b2348__{hab_selection}_{train_split_attempt}_{timeframe}__rstr.tif",
            )
        )
    return paths