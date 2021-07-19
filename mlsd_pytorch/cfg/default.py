from yacs.config import CfgNode as CN
__all_ = ['get_cfg_defaults']
##
_C = CN()
_C.sys = CN()
_C.sys.cpu = False
_C.sys.gpus = 1
_C.sys.num_workers = 8
##
_C.datasets = CN()
_C.datasets.name = ''
_C.datasets.input_size =  512
_C.datasets.with_centermap_extend =  False


##
_C.model = CN()
_C.model.model_name  = ''
_C.model.with_deconv = False
_C.model.num_classes = 1


##
_C.train = CN()
_C.train.do_train = True
_C.train.batch_size = 48
_C.train.save_dir = ''
_C.train.gradient_accumulation_steps = 1
_C.train.num_train_epochs = 170
_C.train.use_step_lr_policy = False
_C.train.warmup_steps = 200
_C.train.learning_rate = 0.0008
_C.train.dropout = 0.1
_C.train.milestones = [100, 150]
_C.train.milestones_in_epo = True
_C.train.lr_decay_gamma = 0.1
_C.train.weight_decay = 0.000001
_C.train.device_ids_str = "0"
_C.train.device_ids = [0]
_C.train.adam_epsilon = 1e-6
_C.train.early_stop_n = 200
_C.train.device_ids_str = "0"
_C.train.device_ids = [0]
_C.train.num_workers = 8
_C.train.log_steps = 50

_C.train.img_dir = ''
_C.train.label_fn = ''
_C.train.data_cache_dir = ''
_C.train.with_cache = False
_C.train.cache_to_mem = False


_C.train.load_from = ""
##
_C.val = CN()
_C.val.batch_size = 8

_C.val.img_dir = ''
_C.val.label_fn = ''

_C.val.val_after_epoch = 0

_C.loss = CN()
_C.loss.loss_weight_dict_list = []
_C.loss.loss_type = '1*L1'
_C.loss.with_sol_loss = True
_C.loss.with_match_loss = False
_C.loss.with_focal_loss = True
_C.loss.match_sap_thresh = 5.0
_C.loss.focal_loss_level = 0

_C.decode = CN()
_C.decode.score_thresh = 0.05
_C.decode.len_thresh = 5
_C.decode.top_k = 500


def get_cfg_defaults(merge_from = None):
  cfg =  _C.clone()
  if merge_from is not None:
      cfg.merge_from_other_cfg(merge_from)
  return cfg