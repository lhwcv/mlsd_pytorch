from  torch.utils.data import  Dataset,DataLoader
from  mlsd_pytorch.data.wireframe_dset import Line_Dataset, LineDataset_collate_fn

__mapping_dataset = {
    'wireframe': Line_Dataset,
}

__mapping_dataset_collate_fn = {
    'wireframe': LineDataset_collate_fn,
}

def get_dataset(cfg, is_train = True):
    if cfg.datasets.name not in __mapping_dataset.keys():
        raise  NotImplementedError('Dataset Type : {} not supported!'.format(cfg.datasets.name))
    return  __mapping_dataset[cfg.datasets.name](
        cfg,
        is_train = is_train
    )

def get_collate_fn(cfg):
    if cfg.datasets.name not in __mapping_dataset_collate_fn.keys():
        raise NotImplementedError('Dataset Type not supported!')
    return __mapping_dataset_collate_fn[cfg.datasets.name]

def get_train_dataloader(cfg):
    ds = get_dataset(cfg, True)
    dloader = DataLoader(
        ds,
        batch_size = cfg.train.batch_size,
        shuffle = True,
        num_workers = cfg.sys.num_workers,
        drop_last=True,
        collate_fn= get_collate_fn(cfg)
    )
    return dloader

def get_val_dataloader(cfg):
    ds = get_dataset(cfg, False)
    dloader = DataLoader(
        ds,
        batch_size = cfg.val.batch_size,
        shuffle = False,
        num_workers = cfg.sys.num_workers,
        drop_last=False,
        collate_fn= get_collate_fn(cfg)
    )
    return dloader

