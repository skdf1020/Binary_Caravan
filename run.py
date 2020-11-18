from omegaconf import MISSING, DictConfig,OmegaConf
import hydra
from pytorch_lightning import Trainer, seed_everything
import transform as _transform, network as networks, pl, dataloader as _dataloader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import utils
import torch
import warnings

warnings.filterwarnings(action='ignore')

# tensorboard --logdir /root/labs/datahouse/NST/pth/NST_effi/ --bind_all

seed_everything(328328)  # seed for reproduciblity


@hydra.main(config_path='./config/main_config.yaml')#, config_name='main_config.yaml')
def my_app(cfg):
    # transform
    train_transform = _transform.get_transform(cfg.transform.name)(cfg.transform.spec, ver=cfg.transform.ver,
                                                                      randaugment=cfg.transform.randaugment,
                                                                      rand_n=cfg.transform.rand_n,
                                                                      rand_m=cfg.transform.rand_m).train_transform
    transform = _transform.get_transform(cfg.transform.name)(cfg.transform.spec, ver=cfg.transform.ver,
                                                                randaugment=cfg.transform.randaugment,
                                                                rand_n=cfg.transform.rand_n,
                                                                rand_m=cfg.transform.rand_m).transform

    # dataloader
    loader_dict = _dataloader.prepare_loader(cfg.data.main_path,
                                                train_transform, transform,
                                                cfg.data.batch_size,
                                                cfg.data.test_batch_size,
                                                cfg.data.num_workers).loader_dict()

    # Network
    net = networks.get_network(cfg.network.net)(ver=cfg.network.ver)
    # onyl for efficient net params record
    OmegaConf.update(cfg, "network.structure", net.structure)

    # model
    model = pl.Lt(net, loader_dict, cfg.opt, cfg.model.loss_func, cfg)

    # logger
    logger = TensorBoardLogger(save_dir=cfg.logger.save_dir,
                               name=cfg.logger.name,
                               version=cfg.logger.ver)

    checkpoint_callback = ModelCheckpoint(
        filepath=Path(cfg.logger.save_dir+'/'+cfg.logger.name+'/'+str(cfg.logger.ver)),
        save_top_k=1,
        verbose=False,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    # trainer
    trainer = Trainer(max_epochs=cfg.trainer.max_epochs,
                      gpus=cfg.trainer.gpus,
                      # weights_save_path=cfg.logger.save_dir,
                      # early_stop_callback = cfg.trainer.callbacks,
                      # distributed_backend=cfg.trainer.distributed,
                      # log_save_interval=cfg.trainer.log_save_interval,
                      checkpoint_callback=checkpoint_callback,
                      logger=logger,
                      auto_lr_find=cfg.trainer.auto_lr
                      )

    #     if cfg.trainer.resume :
    #         trainer = Trainer(max_epochs=cfg.trainer.max_epochs,
    #                   gpus=cfg.trainer.gpus,
    #                   #weights_save_path=cfg.logger.save_dir,
    #                   #early_stop_callback = cfg.trainer.callbacks,
    #                   #distributed_backend=cfg.trainer.distributed,
    #                   log_save_interval=cfg.trainer.log_save_interval,
    #                   logger=logger,
    #                   auto_lr_find=cfg.trainer.auto_lr#,
    #                   #resume_from_checkpoint=cfg.trainer.resume
    #                          )
    # fit & test
    if cfg.trainer.fitting:
        trainer.fit(model)

    print(model.trainer.checkpoint_callback.best_model_path)
    if cfg.trainer.testing:
        trainer.test()
    print(trainer.callback_metrics['avg_test_loss'])
    print(trainer.callback_metrics['avg_test_acc'])

    torch.save(model.network.state_dict(),
               Path(cfg.logger.save_dir+'/'+cfg.logger.name+'/'+str(cfg.logger.ver)+'/best_state.pth'))
               # Path(cfg.logger.save_dir) / Path(cfg.logger.name) / Path(cfg.logger.ver) / 'best_state.pth')
    print(Path(cfg.logger.save_dir+'/'+cfg.logger.name+'/'+str(cfg.logger.ver)+'/best_state.pth'))


if __name__ == '__main__':
    my_app()
