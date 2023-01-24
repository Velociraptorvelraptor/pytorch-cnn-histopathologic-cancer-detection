import pytorch_lightning as pl
from torch import nn, optim
from torchmetrics.functional import accuracy

class CNNImageClassifier(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate

        self.conv_layer1 = nn.Conv2d(3, 3, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv_layer2 = nn.Conv2d(3, 6, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(16 * 16 * 6, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 120)
        self.fc5 = nn.Linear(120, 60)
        self.fc6 = nn.Linear(60, 2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        x = self.conv_layer1(input)
        x = self.relu1(x)
        x = self.pool(x)

        x = self.conv_layer2(x)
        x = self.relu2(x)

        x = x.view(-1, 6 * 16 * 16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        return x

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        train_accuracy = accuracy(outputs, targets, task='multiclass', num_classes=2)
        loss = self.loss(outputs, targets)
        self.log('train_accuracy', train_accuracy, prog_bar=True)
        self.log('train_loss', loss)
        return {'loss': loss, 'train_accuracy': train_accuracy}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        val_accuracy = accuracy(outputs, targets, task='multiclass', num_classes=2)
        loss = self.loss(outputs, targets)
        self.log('val_accuracy', val_accuracy, prog_bar=True)
        self.log('val_loss', loss)
        return {'loss': loss, 'val_accuracy': val_accuracy}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        test_accuracy = accuracy(outputs, targets, task='multiclass', num_classes=2)
        loss = self.loss(outputs, targets)
        self.log('test_accuracy', test_accuracy, prog_bar=True)
        self.log('test_loss', loss)
        return {'loss': loss, 'test_accuracy': test_accuracy}

    def predict_step(self, batch, batch_idx ):
        return self(batch)