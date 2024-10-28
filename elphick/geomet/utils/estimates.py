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

    if show_plot:
        complement_stream: MassComposition = input_stream.sub(estimate_stream, name=complement_name)
        fs: Flowsheet = Flowsheet.from_objects([input_stream, estimate_stream, complement_stream],
                                               name=f"{fs_name}: Balance prior to coercion")
        fs.table_plot(plot_type='network').show()

    # coerce the component mass to within the total dry mass
    estimate_stream = coerce_component_mass(estimate_stream)

    if estimate_stream.status.ok is False:
        logging.info(str(estimate_stream.status.num_oor) + ' records are out of range in the estimate stream.')

    if input_stream.status.ok is False:
        raise ValueError('Input stream is not OK')

    # calculate the recovery
    if estimate_stream.moisture_in_scope:
        recovery: pd.DataFrame = estimate_stream.get_mass_data().drop(
            columns=estimate_stream.mass_wet_var) / input_stream.get_mass_data().drop(columns=input_stream.mass_wet_var)
    else:
        recovery: pd.DataFrame = estimate_stream._mass_data / input_stream._mass_data

    # limit the recovery to the bounds
    recovery = recovery.clip(lower=recovery_bounds[0], upper=recovery_bounds[1])

    # recalculate the estimate from the bound recovery
    if estimate_stream.moisture_in_scope:
        new_mass: pd.DataFrame = recovery * input_stream.get_mass_data()[recovery.columns]
        # calculate wmt and drop h2o, to match what is stored in _mass_data
        new_mass.insert(loc=0, column=estimate_stream.mass_wet_var,
                        value=new_mass[estimate_stream.mass_dry_var] + new_mass[estimate_stream.moisture_column])
        new_mass.drop(columns=estimate_stream.moisture_column, inplace=True)
    else:
        new_mass: pd.DataFrame = recovery * input_stream.mass_data

    estimate_stream.update_mass_data(new_mass)

    if estimate_stream.status.ok is False:
        raise ValueError('Estimate stream is not OK - it should be after bounding recovery')

    # solve the complement
    complement_stream: MassComposition = input_stream.sub(estimate_stream, name=complement_name)
    if complement_stream.status.ok is False:

        # adjust the complement to be within the component
        cols = [col for col in complement_stream.data.columns if col not in [complement_stream.mass_wet_var]]
        new_complement_composition = complement_stream.data[cols]
        for comp, comp_range in complement_stream.status.ranges.items():
            new_complement_composition[comp] = new_complement_composition[comp].clip(comp_range[0], comp_range[1])
        new_component_mass: pd.DataFrame = composition_to_mass(new_complement_composition,
                                                               mass_wet=complement_stream.mass_wet_var,
                                                               mass_dry=complement_stream.mass_dry_var)
        complement_stream.update_mass_data(new_component_mass)

        # adjust the estimate to maintain the balance
        estimate_stream = input_stream.sub(complement_stream, name=estimate_stream.name,
                                           include_supplementary_data=True)

        if estimate_stream.status.ok is False:
            raise ValueError('Estimate stream is not OK after adjustment')

    fs2: Flowsheet = Flowsheet.from_objects([input_stream, estimate_stream, complement_stream],
                                            name=f"{fs_name}: Coerced Estimates")

    if show_plot:
        fs2.table_plot(plot_type='network').show()

    if fs2.all_nodes_healthy is False:
        raise ValueError('Flowsheet is not balanced after adjustment')

    return estimate_stream


def coerce_component_mass(input_stream: Stream):
    """Coerce the component mass to within the total dry mass

    When estimates are made independently, the component mass can exceed the total dry mass.  This function coerces
    the component mass to be within the total dry mass.

    Note that this can modify the grades of the components, and should be used with caution.
    The number and indexes of the coerced records are logged.

    Args:
        input_stream: The input stream

    Returns:

    """

    # calculate the total dry mass
    wet_dry_mass = input_stream.get_mass_data()[[input_stream.mass_wet_var, input_stream.mass_dry_var]]
    total_dry_mass = wet_dry_mass[input_stream.mass_dry_var]

    # calculate the sum of the component masses
    component_mass = input_stream.get_mass_data().drop(columns=[wet_dry_mass.columns])
    sum_component_mass = component_mass.sum(axis=1)

    # calculate the ratio of the total dry mass to the sum of the component masses
    ratio = total_dry_mass / sum_component_mass

    # multiply the component masses by the ratio
    new_component_mass = component_mass * ratio

    # update the mass data
    input_stream.update_mass_data(pd.concat([wet_dry_mass, new_component_mass], axis=1))

    # log the number of records coerced
    num_coerced = sum_component_mass[sum_component_mass > total_dry_mass].count()
    logging.info(f'{num_coerced} records were coerced to maintain the total dry mass.  '
                 f'Indexes: {sum_component_mass[sum_component_mass > total_dry_mass].index}')

    return input_stream
