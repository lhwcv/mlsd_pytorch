import  os
import torch
import math
import sys

sys.path.append(os.path.dirname(__file__)+'/../')

from  mlsd_pytorch.utils.logger import TxtLogger
from  mlsd_pytorch.utils.comm import setup_seed, create_dir
from  mlsd_pytorch.cfg.default import  get_cfg_defaults
from  mlsd_pytorch.optim.lr_scheduler import WarmupMultiStepLR

from  mlsd_pytorch.data import  get_train_dataloader, get_val_dataloader
from  mlsd_pytorch.learner import Simple_MLSD_Learner
from  mlsd_pytorch.models.build_model import  build_model


import  argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default= os.path.dirname(__file__)+ '/configs/mobilev2_mlsd_tiny_512_base.yaml',
                        type=str,
                        help="")
    return parser.parse_args()

def train(cfg):
    train_loader = get_train_dataloader(cfg)
    val_loader   = get_val_dataloader(cfg)
    model = build_model(cfg).cuda()


    #print(model)
    if os.path.exists(cfg.train.load_from):
        print('load from: ', cfg.train.load_from)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(cfg.train.load_from,map_location=device),strict=False)

    if cfg.train.milestones_in_epo:
        ns = len(train_loader)
        milestones = []
        for m in cfg.train.milestones:
            milestones.append(m * ns)
        cfg.train.milestones = milestones

    optimizer = torch.optim.Adam(params=model.parameters(),lr=cfg.train.learning_rate,weight_decay=cfg.train.weight_decay)

    if cfg.train.use_step_lr_policy:

        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones= cfg.train.milestones,
            gamma = cfg.train.lr_decay_gamma,
            warmup_iters=cfg.train.warmup_steps,
        )
    else: ## similiar with in the paper
        warmup_steps = 5 * len(train_loader) ## 5 epoch warmup
        min_lr_scale = 0.0001
        start_step = 70 * len(train_loader)
        end_step = 150 * len(train_loader)
        n_t = 0.5
        lr_lambda_fn = lambda step: (0.9 * step / warmup_steps + 0.1) if step < warmup_steps else \
            1.0 if step < start_step else \
                min_lr_scale if \
                    n_t * (1 + math.cos(math.pi * (step - start_step) / (end_step - start_step))) < min_lr_scale else \
                    n_t * (1 + math.cos(math.pi * (step - start_step) / (end_step - start_step)))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fn)

    create_dir(cfg.train.save_dir)
    logger = TxtLogger(cfg.train.save_dir + "/train_logger.txt")
    learner =  Simple_MLSD_Learner(
        cfg,
        model = model,
        optimizer = optimizer,
        scheduler = lr_scheduler,
        logger = logger,
        save_dir = cfg.train.save_dir,
        log_steps = cfg.train.log_steps,
        device_ids = cfg.train.device_ids,
        gradient_accum_steps = 1,
        max_grad_norm = 1000.0,
        batch_to_model_inputs_fn = None,
        early_stop_n= cfg.train.early_stop_n)

    #learner.val(model, val_loader)
    #learner.val(model, train_loader)
    learner.train(train_loader, val_loader, epoches= cfg.train.num_train_epochs)


if __name__ == '__main__':
    setup_seed(6666)
    cfg = get_cfg_defaults()
    args = get_args()

    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    print('using config: ',args.config.strip())
    cfg.merge_from_file(args.config)
    print(cfg)

    create_dir(cfg.train.save_dir)
    cfg_str = cfg.dump()
    with open(cfg.train.save_dir+ "/cfg.yaml", "w") as f:
        f.write(cfg_str)
    f.close()

    train(cfg)