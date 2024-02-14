import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from diffsynth.layers import Resnet1D, Normalize2d
from diffsynth.transforms import LogTransform
from synth import FMSynth
from nnAudio.Spectrogram import MelSpectrogram, MFCC

#Basic LSTM network, based on the simpler model proposed by MYK

# LSTM() returns tuple of (tensor, (recurrent state))

class unpack_sequence(nn.Module):
    def forward(self, x):
        output, _ = pad_packed_sequence(x, batch_first=True)
        return output

class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        #print(tensor.shape)
        # Reshape shape (batch, hidden)
        return tensor
        
class CustomLSTMModel(nn.Module):
    def __init__(self, input_shape, num_outputs, **kwargs):
        super(CustomLSTMModel, self).__init__()
        
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        
        self.build_model()
        
    def build_model(self, highway_layers=100):
        self.model = nn.Sequential(
            nn.LSTM(input_size=self.input_shape[2], hidden_size=highway_layers, batch_first=True, 
                    num_layers=1, bidirectional=False),
            extract_tensor(),
            nn.LSTM(input_size=highway_layers, hidden_size=highway_layers, batch_first=True, 
                    num_layers=1, bidirectional=False),
            extract_tensor(),
            nn.LSTM(input_size=highway_layers, hidden_size=highway_layers, batch_first=True, 
                    num_layers=1, bidirectional=False),
            extract_tensor(),
            unpack_sequence(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(highway_layers, self.num_outputs),
        )
    
    def forward(self, x):
        # Pack the padded sequence

        #remove last 1 dimension
        #and permute to have order (batch, seq_length, num_features)
        mfcc = torch.squeeze(x['mfcc'], dim=-1)
        mfcc = torch.permute(mfcc, (0, 2, 1))
        sequence_lengths = torch.squeeze(x['mfcc_length'], dim=-1) #same for lengths tensor
        #print(mfcc.shape) #for debugging shape
        packed_seq = pack_padded_sequence(mfcc, sequence_lengths, batch_first=True, enforce_sorted=False)
        
        # Forward pass through the model
        output = self.model(packed_seq)
        
        # Unpack the sequence
        #output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Get the last timestep of each sequence
        last_timestep = torch.arange(output.size(0)) * output.size(1) + sequence_lengths - 1
        last_output = output.view(-1, output.size(2))[last_timestep]
        
        # Build synth controls output dictionary
        controls = {}
        controls['mod_index'] = last_output[:, 0:6]
        controls['freq_ratio'] = last_output[:, 6:12]
        controls['adsr'] = last_output[:, 12:]
        controls['f0'] = x['f0'] #pass along the tracked frequency (computed during data processing)
        return controls


class LSTMPlusSynth(nn.Module):
    def __init__(self, input_shape, num_outputs, audio_length, sr=16000, **kwargs):
        super(LSTMPlusSynth, self).__init__()
        
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.audio_length = audio_length
        self.sr = sr
        
        self.build_model()
        
    def build_model(self, highway_layers=100):
        net = []
        net.append(CustomLSTMModel(self.input_shape, self.num_outputs))
        net.append(FMSynth(self.sr, 64, 0.32, self.audio_length))
        self.model = nn.Sequential(*net)
    
    def forward(self, x):
        out = self.model(x)
        return out

    def get_sr(self):
        return self.sr



#1d Convolutional + GRU model from Masuda paper

class MelEstimator(nn.Module):
    def __init__(self, output_dim, n_mels=128, n_fft=1024, hop=256, sample_rate=16000, channels=64, kernel_size=7, strides=[2,2,2], num_layers=1, hidden_size=512, dropout_p=0.0, bidirectional=False, norm='batch'):
        super().__init__()
        self.n_mels = n_mels
        self.channels = channels
        self.logmel = nn.Sequential(MelSpectrogram(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop, center=True, power=1.0, htk=True, trainable_mel=False, trainable_STFT=False), LogTransform())
        self.norm = Normalize2d(norm) if norm else None
        # Regular Conv
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(1, channels, kernel_size,
                        padding=kernel_size // 2,
                        stride=strides[0]), nn.BatchNorm1d(channels), nn.ReLU())]
            + [nn.Sequential(nn.Conv1d(channels, channels, kernel_size,
                         padding=kernel_size // 2,
                         stride=strides[i]), nn.BatchNorm1d(channels), nn.ReLU())
                         for i in range(1, len(strides))])
        self.l_out = self.get_downsampled_length()[-1] # downsampled in frequency dimension
        print('output dims after convolution', self.l_out)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.gru = nn.GRU(self.l_out * channels, hidden_size, num_layers=num_layers, dropout=dropout_p, batch_first=True, bidirectional=bidirectional)
        self.out = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_dim)
        self.output_dim = output_dim

    def forward(self, input):
        audio = input['audio'] #clean this up later
        audio = audio.permute(0, 2, 1) #needed for nnAudio analysis libraries (mel spectrogram)
        x = self.logmel(audio)
        x = self.norm(x)
        batch_size, n_mels, n_frames = x.shape
        print(batch_size, n_mels, n_frames)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, self.n_mels).unsqueeze(1)
        # x: [batch_size*n_frames, 1, n_mels]
        for i, conv in enumerate(self.convs):
            x = conv(x)
        x = x.view(batch_size, n_frames, self.channels, self.l_out)
        x = x.view(batch_size, n_frames, -1)
        D = 2 if self.bidirectional else 1
        output, _hidden = self.gru(x, torch.zeros(D * self.num_layers, batch_size, self.hidden_size, device=x.device))
        # output: [batch_size, n_frames, self.output_dim]
        output = self.out(output)
        output = torch.sigmoid(output)

        #Only take the last one
        last_output = output[:, -1, :]
        
        #Create output dict
        controls = {}
        controls['mod_index'] = last_output[:, 0:6]
        controls['freq_ratio'] = last_output[:, 6:12]
        controls['adsr'] = last_output[:, 12:]
        controls['f0'] = input['f0'] #pass along the tracked frequency (computed during data processing)
        return controls
        

    def get_downsampled_length(self):
        l = self.n_mels
        lengths = [l]
        for conv in self.convs:
            conv_module = conv[0]
            l = (l + 2 * conv_module.padding[0] - conv_module.dilation[0] * (conv_module.kernel_size[0] - 1) - 1) // conv_module.stride[0] + 1
            lengths.append(l)
        return lengths


class MelPlusSynth(nn.Module):
    def __init__(self, num_outputs, audio_length, sr=16000):
        super(MelPlusSynth, self).__init__()
        self.num_outputs = num_outputs
        self.sr = sr
        self.audio_length = audio_length
        self.build_model()
        
    def build_model(self):
        net = []
        net.append(MelEstimator(output_dim=self.num_outputs))
        net.append(FMSynth(self.sr, 64, 0.32, self.audio_length))
        self.model = nn.Sequential(*net)
    
    def forward(self, x):
        out = self.model(x)
        return out

    def get_sr(self):
        return self.sr
        



