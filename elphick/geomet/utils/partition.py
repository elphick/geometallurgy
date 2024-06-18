import numpy as np
import pandas as pd


def perfect(x: np.ndarray, d50: float) -> np.ndarray:
    """A perfect partition
    
    Args:
        x: The input dimension, e.g. size or density
        d50: The cut-point

    Returns:

    """
    pn: np.ndarray = np.where(x >= d50, 100.0, 0.0)
    return pn


def napier_munn(x: np.ndarray, d50: float, ep: float) -> np.ndarray:
    """The Napier-Munn partition (1998)

    REF: https://www.sciencedirect.com/science/article/pii/S1474667016453036

    Args:
        x: The input dimension, e.g. size or density
        d50: The cut-point
        ep: The Escarte Probable

    Returns:

    """
    pn: np.ndarray = 1 / (1 + np.exp(1.099 * (d50 - x) / ep)) * 100
    return pn


# if __name__ == '__main__':
#     da = np.arange(0, 10)
#     PN = perfect(da, d50=6.3)
#     df = pd.DataFrame([da, PN], index=['da', 'pn']).T
#     print(df)
#
#     da = np.arange(0, 10)
#     PN = napier_munn(da, d50=6.3, ep=0.1)
#     df = pd.DataFrame([da, PN], index=['da', 'pn']).T
#     print(df)
