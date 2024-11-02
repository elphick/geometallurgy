import logging

import pandas as pd

from elphick.geomet.base import MassComposition
from elphick.geomet.flowsheet import Flowsheet
from elphick.geomet.flowsheet.stream import Stream
from elphick.geomet.utils.moisture import solve_mass_moisture
from elphick.geomet.utils.pandas import composition_to_mass


def coerce_estimates(estimate_stream: Stream, input_stream: Stream,
                     recovery_bounds: tuple[float, float] = (0.01, 0.99),
                     complement_name: str = 'complement',
                     fs_name: str = 'Flowsheet',
                     show_plot: bool = False) -> Stream:
    """Coerce output estimates within recovery and the component range.

    Estimates contain error and at times can exceed the specified component range, or can consume more component
    mass than is available in the feed.  This function modifies (coerces) only the non-compliant estimate records
    in order to balance the node and keep all dry components within range.  Moisture is not modified.

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
        complement_name: The name of the complement , for plots.
        fs_name: The name of the flowsheet, for plots.
        show_plot: If True, show the network plot

    Returns:
        The coerced estimate stream
    """

    if show_plot:
        complement_stream: MassComposition = input_stream.sub(estimate_stream, name=complement_name)
        fs: Flowsheet = Flowsheet.from_objects([input_stream, estimate_stream, complement_stream],
                                               name=f"{fs_name}: Balance prior to coercion")
        fs.table_plot(plot_type='network', table_area=0.2, table_pos='top').show()

    # clip the composition to the bounds
    estimate_stream = estimate_stream.clip_composition()

    # coerce the estimate component mass to within the total dry mass
    estimate_stream = estimate_stream.balance_composition()

    # store the original dry mass, to be used in the moisture calculation
    original_mass_moisture: pd.DataFrame = pd.DataFrame()
    if estimate_stream.moisture_in_scope:
        original_mass_moisture: pd.DataFrame = estimate_stream.get_mass_data(include_moisture=True)[
            estimate_stream.mass_columns + [estimate_stream.moisture_column]]

    if estimate_stream.status.ok is False:
        logging.info(str(estimate_stream.status.num_oor) + f' records are out of range in the estimate stream '
                                                           f'{estimate_stream.name}.')

    if input_stream.status.ok is False:
        raise ValueError('Input stream is not OK')

    # calculate the recovery
    recovery: pd.DataFrame = (estimate_stream.get_mass_data(include_moisture=False) /
                              input_stream.get_mass_data(include_moisture=False))

    # limit the recovery to the bounds
    recovery = recovery.clip(lower=recovery_bounds[0], upper=recovery_bounds[1]).fillna(0.0)

    # Recalculate the estimate from the bound recovery
    new_mass: pd.DataFrame = recovery * input_stream.get_mass_data(include_moisture=False)[recovery.columns]

    # if the dry mass has changed the moisture would change since the wet mass remains the same
    if estimate_stream.moisture_in_scope:
        # recalculate the wet mass for the dry mass records that have changed
        changed_record_indexes = new_mass[
            new_mass[estimate_stream.mass_dry_var] < original_mass_moisture[estimate_stream.mass_dry_var]].index
        new_mass_dry: pd.Series = new_mass.loc[changed_record_indexes, estimate_stream.mass_dry_var]
        original_moisture: pd.Series = original_mass_moisture.loc[
            changed_record_indexes, estimate_stream.moisture_column]
        new_wet_mass: pd.Series = solve_mass_moisture(mass_dry=new_mass_dry, moisture=original_moisture)
        new_mass.loc[changed_record_indexes, estimate_stream.mass_wet_var] = new_wet_mass

    estimate_stream.update_mass_data(new_mass)

    if estimate_stream.status.ok is False:
        raise ValueError('Estimate stream is not OK - it should be after bounding recovery')

    # solve the complement
    complement_stream: Stream = input_stream.sub(estimate_stream, name=complement_name)

    # coerce the complement component mass to within the total dry mass
    complement_stream = complement_stream.balance_composition()

    if complement_stream.status.ok is False:

        # clip the composition to the bounds
        complement_stream = complement_stream.clip_composition()

        # coerce the estimate component mass to within the total dry mass
        complement_stream = complement_stream.balance_composition()

        # adjust the estimate to maintain the balance
        estimate_stream = input_stream.sub(complement_stream, name=estimate_stream.name,
                                           include_supplementary_data=True)

        if estimate_stream.status.ok is False:
            raise ValueError('Estimate stream is not OK after adjustment')

    fs2: Flowsheet = Flowsheet.from_objects([input_stream, estimate_stream, complement_stream],
                                            name=f"{fs_name}: Coerced Estimates")

    if show_plot:
        fs2.table_plot(plot_type='network', table_area=0.2, table_pos='top').show()

    if fs2.all_nodes_healthy is False:
        if fs2.all_streams_healthy and not fs2.all_nodes_healthy:
            logging.warning('All streams are healthy but not all nodes are healthy.  Consider the water balance.')
        else:
            raise ValueError('Flowsheet is not balanced after adjustment')

    return estimate_stream
