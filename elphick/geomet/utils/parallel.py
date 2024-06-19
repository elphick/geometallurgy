from joblib import Parallel
from tqdm import tqdm


class TqdmParallel(Parallel):
    def __init__(self, *args, **kwargs):
        self._desc = kwargs.pop('desc', None)  # Get the description from kwargs
        self._tqdm = tqdm(total=kwargs.pop('total', None), desc=self._desc)  # Pass the description to tqdm
        super().__init__(*args, **kwargs)

    def __call__(self, iterable):
        iterable = list(iterable)
        self._tqdm.total = len(iterable)
        result = super().__call__(iterable)
        self._tqdm.close()
        return result

    def _print(self, msg, *msg_args):
        return

    def print_progress(self):
        self._tqdm.update()

    def _dispatch(self, batch):
        job_idx = super()._dispatch(batch)
        return job_idx

    def _collect(self, output):
        return super()._collect(output)
