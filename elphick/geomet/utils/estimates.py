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

        # # debugging snippet to show a failing record
        # qry: str = 'index==1000'
        # fs_debug: Flowsheet = fs.query(qry)
        # fs_debug.name = f"{fs_name}: Balance prior to coercion: [{qry}]"
        # fs_debug.table_plot(plot_type='network', table_area=0.2, table_pos='top').show()

    if input_stream.status.ok is False:
        raise ValueError('Input stream is not OK')

    # clip the composition to the bounds
    estimate_stream = estimate_stream.clip_composition()

    # coerce the estimate component mass to within the total dry mass
    estimate_stream = estimate_stream.balance_composition()

    # clip the recovery
    estimate_stream = estimate_stream.clip_recovery(recovery_bounds=(0.01, 0.99))

    if estimate_stream.status.ok is False:
        raise ValueError('Estimate stream is not OK - it should be after bounding recovery')

    # solve the complement
    complement_stream: Stream = input_stream.sub(estimate_stream, name=complement_name)

    # clip the composition to the bounds
    complement_stream = complement_stream.clip_composition()

    # coerce the estimate component mass to within the total dry mass
    complement_stream = complement_stream.balance_composition()

    # adjust the estimate to maintain the balance
    estimate_stream = input_stream.sub(complement_stream, name=estimate_stream.name,
                                       include_supplementary_data=True)

    if estimate_stream.status.ok is False:
        # This can occur in cases where the complement grade has been reduced (by balance_composition) to a point
        # where the resultant estimate grade is out of range.  In this case, we need to adjust the complement grade.

        estimate_stream = estimate_stream.clip_composition()
        complement_stream = input_stream.sub(estimate_stream, name=complement_name)

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
