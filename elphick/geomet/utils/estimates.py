import pandas as pd

from elphick.geomet.base import MassComposition
from elphick.geomet.flowsheet import Flowsheet
from elphick.geomet.utils.pandas import composition_to_mass


def coerce_output_estimates(estimate_stream: MassComposition, input_stream: MassComposition,
                            recovery_bounds: tuple[float, float] = (0.01, 0.99)) -> MassComposition:
    """Coerce output estimates within recovery and the component range.

    Estimates contain error and at times can exceed the specified component range, or can consume more component
    mass than is available in the feed.  This function:solves the balance assuming one output complement stream,
    1. limits the estimate to within the recovery bounds,
    2. ensures the estimate is within the component range,
    3. solves the complement, and ensures it is in range,
    4. if the complement is out of range, it is adjusted and the estimate adjusted to maintain the balance.

    Args:
        estimate_stream: The estimated object, which is a node output
        input_stream: The input object, which is a node input
        recovery_bounds: The bounds for the recovery, default is 0.01 to 0.99

    Returns:
        The coerced estimate stream
    """

    if input_stream.status.ok is False:
        raise ValueError('Input stream is not OK')

    # calculate the recovery
    cols: list[str] = [estimate_stream.mass_dry_var] + estimate_stream.composition_columns
    recovery: pd.DataFrame = estimate_stream.mass_data[cols] / input_stream.mass_data[cols]

    # limit the recovery to the bounds
    recovery = recovery.clip(lower=recovery_bounds[0], upper=recovery_bounds[1])

    # recalculate the estimate from the bound recovery
    new_mass: pd.DataFrame = recovery * input_stream.mass_data[cols]
    estimate_stream.update_mass_data(new_mass)

    if estimate_stream.status.ok is False:
        raise ValueError('Estimate stream is not OK - it should be after bounding recovery')

    # solve the complement
    complement_stream: MassComposition = input_stream.sub(estimate_stream, name='complement')
    if complement_stream.status.ok is False:

        # adjust the complement to be within the component without creating a new method
        new_complement_composition = complement_stream.data[complement_stream.composition_columns]
        for comp, comp_range in complement_stream.status.ranges.items():
            new_complement_composition[comp] = new_complement_composition[comp].clip(comp_range[0], comp_range[1])
        new_component_mass: pd.DataFrame = composition_to_mass(new_complement_composition,
                                                               mass_dry=complement_stream.mass_dry_var)
        complement_stream.update_mass_data(new_component_mass)

        # adjust the estimate to maintain the balance
        estimate_stream = input_stream.sub(complement_stream, name=estimate_stream.name,
                                           include_supplementary_data=True)

        if estimate_stream.status.ok is False:
            raise ValueError('Estimate stream is not OK after adjustment')

    fs: Flowsheet = Flowsheet.from_objects([input_stream, estimate_stream, complement_stream])
    if fs.balanced is False:
        raise ValueError('Flowsheet is not balanced after adjustment')

    # fs.plot_network().show()

    return estimate_stream
