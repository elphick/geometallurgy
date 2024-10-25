import logging

import pandas as pd

from elphick.geomet.base import MassComposition
from elphick.geomet.flowsheet import Flowsheet
from elphick.geomet.flowsheet.stream import Stream
from elphick.geomet.utils.pandas import composition_to_mass


def coerce_output_estimates(estimate_stream: Stream, input_stream: Stream,
                            recovery_bounds: tuple[float, float] = (0.01, 0.99),
                            complement_name: str = 'complement',
                            show_plot: bool = False) -> Stream:
    """Coerce output estimates within recovery and the component range.

    Estimates contain error and at times can exceed the specified component range, or can consume more component
    mass than is available in the feed.  This function modifies (coerces) only the non-compliant estimate records
    in order to balance the node and keep all components within range.

            estimate_stream (supplied)
                /
    input_stream (supplied)
                \
            complement_stream

    1. limits the estimate to within the recovery bounds,
    2. ensures the estimate is within the component range,
    3. solves the complement, and ensures it is in range,
    4. if the complement is out of range, it is adjusted and the estimate adjusted to maintain the balance.

    Args:
        estimate_stream: The estimated object, which is a node output
        input_stream: The input object, which is a node input
        recovery_bounds: The bounds for the recovery, default is 0.01 to 0.99
        complement_name: The name of the complement stream
        show_plot: If True, show the network plot

    Returns:
        The coerced estimate stream
    """

    if estimate_stream.status.ok is False:
        logging.info(str(estimate_stream.status))

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
    complement_stream: MassComposition = input_stream.sub(estimate_stream, name=complement_name)
    if complement_stream.status.ok is False:

        # adjust the complement to be within the component
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

    if show_plot:
        fs.table_plot(plot_type='network').show()

    if fs.balanced is False:
        raise ValueError('Flowsheet is not balanced after adjustment')

    return estimate_stream


def coerce_input_estimates(estimate_stream: Stream, input_stream: Stream,
                           recovery_bounds: tuple[float, float] = (0.01, 0.99),
                           output_name: str = 'output',
                           show_plot: bool = False) -> Stream:
    """Coerce input estimates within recovery and the component range.

    Estimates contain error and at times can exceed the specified component range, or can consume more component
    mass than is available in the feed.  This function modifies (coerces) only the non-compliant estimate records
    in order to balance the node and keep all components within range.

    estimate_stream (supplied)
               \
                + ──> output_stream
               /
    input_stream (supplied)


    1. limits the estimate to within the recovery bounds,
    2. ensures the estimate is within the component range,
    3. solves the output, and ensures it is in range,
    4. if the output is out of range, it is adjusted and the estimate adjusted to maintain the balance.

    Args:
        estimate_stream: The estimated object, which is a node output
        input_stream: The input object, which is a node input
        recovery_bounds: The bounds for the recovery, default is 0.01 to 0.99
        output_name: The name of the output stream
        show_plot: If True, show the network plot

    Returns:
        The coerced estimate stream
    """

    if estimate_stream.status.ok is False:
        logging.info(str(estimate_stream.status))

    if input_stream.status.ok is False:
        raise ValueError('Input stream is not OK')

    output_stream: Stream = input_stream.add(estimate_stream, name=output_name)

    fs: Flowsheet = Flowsheet.from_objects([input_stream, estimate_stream, output_stream])

    if show_plot:
        fig = fs.table_plot(plot_type='network')
        fig.update_layout(title=f"{fs.name}: Balanced={fs.balanced}").show()

    # calculate the recovery of the estimate relative to the output
    cols: list[str] = [estimate_stream.mass_dry_var] + estimate_stream.composition_columns
    recovery: pd.DataFrame = estimate_stream.mass_data[cols] / output_stream.mass_data[cols]

    # limit the recovery to the bounds
    recovery = recovery.clip(lower=recovery_bounds[0], upper=recovery_bounds[1])

    # recalculate the estimate from the bound recovery
    new_mass: pd.DataFrame = recovery * output_stream.mass_data[cols]
    estimate_stream.update_mass_data(new_mass)

    if estimate_stream.status.ok is False:
        raise ValueError('Estimate stream is not OK - it should be after bounding recovery')

    # re-calculate the output
    output_stream: Stream = input_stream.add(estimate_stream, name=output_name)
    if output_stream.status.ok is False:

        # adjust the complement to be within the component without creating a new method
        new_output_composition = output_stream.data[output_stream.composition_columns]
        for comp, comp_range in output_stream.status.ranges.items():
            new_output_composition[comp] = new_output_composition[comp].clip(comp_range[0], comp_range[1])
        new_component_mass: pd.DataFrame = composition_to_mass(new_output_composition,
                                                               mass_dry=output_stream.mass_dry_var)
        output_stream.update_mass_data(new_component_mass)

        # adjust the estimate to maintain the balance
        estimate_stream = output_stream.sub(input_stream, name=estimate_stream.name,
                                            include_supplementary_data=True)

        if estimate_stream.status.ok is False:
            raise ValueError('Estimate stream is not OK after adjustment')

    fs_out: Flowsheet = Flowsheet.from_objects([input_stream, estimate_stream, output_stream])

    if show_plot:
        fig = fs_out.table_plot(plot_type='network')
        fig.update_layout(title=f"Coerced {fs_out.name}: Balanced={fs_out.balanced}").show()

    if fs_out.balanced is False:
        raise ValueError('Flowsheet is not balanced after adjustment')

    return estimate_stream
