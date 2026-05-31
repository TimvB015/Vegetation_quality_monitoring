from __future__ import annotations

################################################################################
## PATH BUILDER FUNCTION
################################################################################
def path_func(base_dir: Path | str, *parts: str, create: bool = False) -> Path:
    """
    Build and return an absolute path from a base directory and additional path parts.

    The path is formed by joining ``base_dir`` with ``*parts`` and then resolving it
    to an absolute path. If ``create`` is True, the resulting directory is created
    (including parents) if it does not already exist.

    :param base_dir: Base directory to start from (as a ``Path`` or string).
    :type base_dir: Path | str
    :param parts: Additional path components to append to ``base_dir``.
    :type parts: str
    :param create: If True, create the resulting directory with
                   ``parents=True`` and ``exist_ok=True``.
    :type create: bool
    :return: The resolved absolute path.
    :rtype: Path
    :raises OSError: If directory creation is requested and fails due to OS-level errors
                     (e.g., permissions, invalid path).
    """
    base = Path(base_dir)
    p = (base.joinpath(*parts)).resolve()
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p



################################################################################
## BATCH FUNTION
################################################################################
def batch_func(years, template, globals_dict):
    """
    Iterates over years, dynamically formatting and executing code templates.

    Args:
        years (list[int]): Years to iterate over.
        template (str): String with placeholders '{year}' for dynamic code.
        globals_dict (dict): Usually 'globals()', to resolve variable names.

    Returns:
        None
    """
    for year in years:
        exec(template.format(year=year), globals_dict)



################################################################################
## IMPORT XLSX
################################################################################
import pandas as pd

def import_xlsx_func(xlsx_path, set_index_col=None, reindex=None, cols_to_keep=None, sheet_name=0):
    """
    Load a DataFrame from the given XLSX path.
    - If set_index_col is provided, set that column as the index after loading.
    - If a column named 'index' exists (and set_index_col is None), set it as the index after loading.
    - If the existing index is already named 'index', keep it.
    - If cols_to_keep is provided, keep only those columns.
    - If reindex is provided (e.g., 'feature'), set a new index like 'feature_1', 'feature_2', ...
      If reindex is None (default), keep the index derived above.
    - sheet_name can be a name or integer, default is the first sheet.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    if set_index_col is not None and set_index_col in df.columns:
        df = df.set_index(set_index_col, drop=True)
    elif 'index' in df.columns:
        df = df.set_index('index', drop=True)
    else:
        if df.index.name != 'index':
            df.index.name = 'index'

    if cols_to_keep is not None:
        cols = [c for c in cols_to_keep if c in df.columns]
        df = df[cols]

    if reindex is not None:
        df.index = [f"{reindex}_{i+1}" for i in range(len(df))]
        df.index.name = 'index'

    return df