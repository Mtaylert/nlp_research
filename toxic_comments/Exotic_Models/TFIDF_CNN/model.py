from torch import nn


class CNNTFIDF(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, num_labels):
        super(CNNTFIDF, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden1, kernel_size=1),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(
                in_channels=hidden1,
                out_channels=hidden2,
                kernel_size=1,
            ),
            nn.ELU(),
            nn.Dropout(p=0.5),
        )

        self.fc = nn.Linear(hidden2, num_labels)

    def forward(self, x):
        features = self.convnet(x)
        features = features.squeeze(dim=2)
        prediction = self.fc(features)
        return prediction
