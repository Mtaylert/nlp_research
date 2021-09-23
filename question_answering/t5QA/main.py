import pytorch_lightning as pl
import config
import data_reader
from data_module import QADataModule
from model import QAModel
from pytorch_lightning.callbacks import ModelCheckpoint


def run(train=False):

    train_df = data_reader.unpack_data(config.TRAIN_DATA)
    val_df =  data_reader.unpack_data(config.VAL_DATA)

    data_module = QADataModule(train_df,val_df)
    data_module.setup()


    model = QAModel()
    checkpoint_callback = ModelCheckpoint(

        dirpath = config.MODEL_PATH,
        filename = 'best-checkpoint_QA',
        save_top_k=1,
        verbose =True,
        monitor = 'val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
      checkpoint_callback=True,
        callbacks = checkpoint_callback,
        max_epochs = config.EPOCHS,
        accelerator='ddp_cpu',
        progress_bar_refresh_rate=30
    )

    trainer.fit(model, data_module)

if __name__ == '__main__':
    run()