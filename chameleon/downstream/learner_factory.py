from .davis2017.learner import DAVIS2017Learner
from .ap10k.learner import AP10KLearner
from .isic2018.learner import ISIC2018Learner
from .cellpose.learner import CELLPOSELearner
from .fsc147.learner import FSC147Learner
# [AIC200 PATCH] student-defined tasks; see downstream/custom/ and chameleon/PATCHES.md
from .custom.registry import get_custom_learner_cls
# [AIC200 PATCH] flat images/ + labels/ demo layout; see downstream/demo.py
from .demo import DEMO_LEARNERS, is_flat_layout


def get_downstream_learner(config, trainer):
    '''
    add custom learner here
    '''
    # [AIC200 PATCH] the demo subsets use one flat layout for every task
    if config.dataset in DEMO_LEARNERS and is_flat_layout(config.path_dict[config.dataset]):
        return DEMO_LEARNERS[config.dataset](config, trainer)
    if config.dataset == 'davis2017':
        return DAVIS2017Learner(config, trainer)
    elif config.dataset == 'ap10k':
        return AP10KLearner(config, trainer)
    elif config.dataset == 'isic2018':
        return ISIC2018Learner(config, trainer)
    elif config.dataset == 'cellpose':
        return CELLPOSELearner(config, trainer)
    elif config.dataset == 'fsc147':
        return FSC147Learner(config, trainer)
    # [AIC200 PATCH]
    elif config.dataset == 'custom':
        return get_custom_learner_cls()(config, trainer)
    else:
        raise NotImplementedError