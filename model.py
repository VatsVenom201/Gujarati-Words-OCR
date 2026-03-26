import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(CRNN, self).__init__()
        
        # CNN Feature Extractor
        # Input: (B, 1, 32, W)
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # (B, 64, 16, W/2)
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # (B, 128, 8, W/4)
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # (B, 256, 4, W/4)
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # (B, 512, 2, W/4)
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)) # (B, 512, 1, W/4)
        )
        
        # RNN Sequence Learner
        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden_size, 
                           bidirectional=True, batch_first=True)
                           
        self.dropout = nn.Dropout(p=0.3)
                           
        # Linear Classifier
        self.linear = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, x):
        # x is (B, 1, 32, W)
        conv = self.cnn(x) # (B, 512, 1, W/4)
        
        b, c, h, w = conv.size()
        # Constraint check: H must be 1
        assert h == 1, "Expected height to be 1"
        
        # Sequence conversion: (B, 512, 1, W/4) -> (B, W/4, 512)
        conv = conv.squeeze(2) # (B, 512, W/4)
        conv = conv.permute(0, 2, 1) # (B, W/4, 512)
        
        # RNN
        rnn_out, _ = self.rnn(conv) # (B, W/4, hidden_size*2)
        
        # Dropout regularization
        rnn_out = self.dropout(rnn_out)
        
        # Linear
        output = self.linear(rnn_out) # (B, W/4, vocab_size)
        
        return output
