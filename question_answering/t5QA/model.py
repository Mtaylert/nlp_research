import pytorch_lightning as pl
import config
import transformers


class QAModel(pl.LightningModule):
    def __init__(self):
        super(QAModel, self).__init__()

        self.model = transformers.T5ForConditionalGeneration.from_pretrained(
            config.T5_PATH, return_dict=True
        )

        print(self.model.config)

    def forward(self, input_ids, attention_mask, labels=None):

        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss,outputs = self(input_ids,attention_mask,labels)
        self.log("train_loss",loss,prog_bar=True,logger=True)

        return loss

    def validation_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss,outputs = self(input_ids,attention_mask,labels)
        self.log("val_loss",loss,prog_bar=True,logger=True)

        return loss


    def test_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss,outputs = self(input_ids,attention_mask,labels)
        self.log("test_loss",loss,prog_bar=True,logger=True)

        return loss


    def configure_optimizers(self):

        return transformers.AdamW(self.parameters(), lr=0.0001)



if __name__ == '__main__':

    model = QAModel()


