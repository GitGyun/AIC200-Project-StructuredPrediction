from .chameleon import Chameleon 
from meta_train.unified import Unified
from downstream.davis2017.dataset import DAVIS2017
# [AIC200 PATCH] student-defined tasks; see downstream/custom/ and chameleon/PATCHES.md
from downstream.custom.registry import get_custom_n_tasks


def get_model(config, verbose=False):
    # set number of tasks for bitfit
    if config.stage == 0:
        n_tasks = len(Unified.TASKS)
        n_task_groups = len(Unified.TASK_GROUP_NAMES)
    else:
        if config.dataset == 'ap10k':
            n_tasks = 17 # number of joints
        elif config.dataset == 'davis2017':
            n_tasks = DAVIS2017.NUM_INSTANCES[DAVIS2017.CLASS_NAMES.index(config.class_name)] # number of instances
        elif config.dataset == 'fsc147':
            n_tasks = 1 # density map
        elif config.dataset == 'cellpose':
            n_tasks = 3 # u, v, segmentation
        # [AIC200 PATCH] number of label channels registered by register_custom(...)
        elif config.dataset == 'custom':
            n_tasks = getattr(config, 'n_tasks', None) or get_custom_n_tasks()
        else:
            n_tasks = 1
        n_task_groups = 1

    model = Chameleon(config, n_tasks, n_task_groups)

    if verbose:
        print(f'Registered Chameleon with {n_tasks} task-specific and {n_task_groups} group-specific parameters.')

    return model
