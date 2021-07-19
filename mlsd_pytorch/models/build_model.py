import  sys
sys.path.append('../../')

from mlsd_pytorch.models.mbv2_mlsd import MobileV2_MLSD
from mlsd_pytorch.models.mbv2_mlsd_large import  MobileV2_MLSD_Large


def build_model(cfg):
    model_name = cfg.model.model_name
    if model_name == 'mobilev2_mlsd':
        m = MobileV2_MLSD(cfg)
        return m
    if model_name == 'mobilev2_mlsd_large':
        m = MobileV2_MLSD_Large(cfg)
        return m
    raise  NotImplementedError('{} no such model!'.format(model_name))
