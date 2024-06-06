"""
Managing components/composition
"""

from typing import List, Dict, Union

import periodictable as pt
from periodictable.formulas import Formula

custom_components: List[str] = ['LOI']

# Kudos: pyrolite
DEFAULT_CHARGES: Dict = dict(
    H=1,
    Li=1,
    Be=1,
    B=3,
    C=4,
    O=-2,
    F=-1,
    Na=1,
    Mg=2,
    Al=3,
    Si=4,
    P=3,
    Cl=-1,
    K=1,
    Ca=2,
    Sc=3,
    Ti=4,
    V=3,
    Cr=3,
    Mn=2,
    Fe=2,
    Co=2,
    Ni=2,
    Cu=2,
    Zn=2,
    Br=-1,
    Rb=1,
    Sr=2,
    Y=3,
    Zr=4,
    Nb=5,
    Sn=4,
    I=-1,
    Cs=1,
    Ba=2,
    La=3,
    Ce=3,
    Pr=3,
    Nd=3,
    Sm=3,
    Eu=3,
    Gd=3,
    Tb=3,
    Dy=3,
    Ho=3,
    Er=3,
    Tm=3,
    Yb=3,
    Lu=3,
    Hf=4,
    Pb=2,
    Th=4,
    U=4,
)


def elements() -> List[str]:
    res: List[str] = [el.symbol for el in pt.elements]
    return res


def is_element(candidates: List[str], strict: bool = True) -> Union[List[str], Dict[str, str]]:
    if strict:
        matches: list = list(set(candidates).intersection(elements()))
    else:
        e_map: Dict[str, str] = {e.symbol.lower(): e.symbol for e in pt.elements}
        matches: Dict[str, str] = {c: e_map[c.lower()] for c in candidates if c.lower() in e_map.keys()}

    return matches


def oxides() -> List[Formula]:
    # cats = {e for e in [el for el in pt.elements if str(el) in DEFAULT_CHARGES.keys()] if DEFAULT_CHARGES[str(e)] > 0}
    cats = {el for el in pt.elements if (str(el) in DEFAULT_CHARGES.keys()) and (DEFAULT_CHARGES[str(el)] > 0)}

    res: List[Formula] = []
    for c in cats:
        charge = DEFAULT_CHARGES[str(c)]
        if charge % 2 == 0:
            res.append(pt.formula(str(c) + str(1) + 'O' + str(charge // 2)))
        else:
            res.append(pt.formula(str(c) + str(2) + 'O' + str(charge)))

    return res


def is_oxide(candidates: List[str], strict: bool = True) -> Union[List[str], Dict[str, str]]:
    if strict:
        oxs = {str(o) for o in oxides()}
        matches: list = list(set(candidates).intersection(oxs))
    else:
        o_map: Dict[str, str] = {str(o).lower(): str(o) for o in oxides()}
        matches: Dict[str, str] = {c: o_map[c.lower()] for c in candidates if c.lower() in o_map.keys()}

    return matches


def is_compositional(candidates: List[str], strict: bool = True) -> Union[List[str], Dict[str, str]]:
    """
    Check if a list of candidates are compositional components (elements or oxides)
    Args:
        candidates: list of string candidates
        strict: If True, the candidates must be in the list of known compositional components (elements or oxides)
        as chemical symbols.

    Returns:
        If strict, a list of compositional components, otherwise a dict of the original candidates (keys) and
        their compositional component symbols (values)
    """
    if strict:
        comps = {str(o) for o in oxides()}.union(set(elements())).union(set(custom_components))
        matches: list = list(set(candidates).intersection(comps))
    else:
        comp_map: Dict[str, str] = {**{str(o).lower(): str(o) for o in oxides()},
                                    **{a.lower(): a for a in elements()},
                                    **{c.lower(): c for c in custom_components}}
        matches: Dict[str, str] = {c: comp_map[c.lower()] for c in candidates if c.lower() in comp_map.keys()}

    return matches


def get_components(candidates: List[str], strict: bool = True) -> list[str]:
    return list(is_compositional(candidates, strict=strict).keys())
