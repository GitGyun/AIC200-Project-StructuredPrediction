"""Registration hook for student-defined tasks.

Chameleon decides three things from ``config.dataset``:

1. which ``torch.utils.data.Dataset`` produces ``(X, Y, M)`` episodes,
2. which ``BaseLearner`` defines the loss / post-processing / visualisation, and
3. how many *task-specific* parameter sets the model allocates (``n_tasks``), which must equal
   the number of channels in your label tensor ``Y``.

For the five built-in benchmarks those are hard-coded in ``downstream/learner_factory.py`` and
``model/model_factory.py``. For the ``custom`` dataset they are looked up here instead, so you can
swap in your own classes from a notebook cell without editing the library::

    from downstream.custom import register_custom, CustomDenseDataset, CustomLearner

    class MyDataset(CustomDenseDataset):
        ...

    register_custom(dataset_cls=MyDataset, learner_cls=CustomLearner, n_tasks=1)

Call ``register_custom`` *before* building the config, because ``n_tasks`` is read when the model
is constructed.
"""

_REGISTRY = {
    'dataset_cls': None,
    'learner_cls': None,
    'n_tasks': 1,
}


def register_custom(dataset_cls=None, learner_cls=None, n_tasks=None):
    """Register the Dataset / Learner / channel count used by ``--dataset custom``.

    Any argument left as ``None`` keeps its previous value, so you can re-register just the
    dataset while iterating in a notebook.
    """
    if dataset_cls is not None:
        _REGISTRY['dataset_cls'] = dataset_cls
    if learner_cls is not None:
        _REGISTRY['learner_cls'] = learner_cls
    if n_tasks is not None:
        if n_tasks < 1:
            raise ValueError(f'n_tasks must be >= 1, got {n_tasks}')
        _REGISTRY['n_tasks'] = n_tasks
    return dict(_REGISTRY)


def get_custom_dataset_cls():
    from .dataset import CustomDenseDataset
    return _REGISTRY['dataset_cls'] or CustomDenseDataset


def get_custom_learner_cls():
    from .learner import CustomLearner
    return _REGISTRY['learner_cls'] or CustomLearner


def get_custom_n_tasks():
    return _REGISTRY['n_tasks']
