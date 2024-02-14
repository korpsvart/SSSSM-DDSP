import torch
import torch.nn as nn
import math
import numpy as np
import adsr as adsr_lib

#Synth definitions

OP6=5
OP5=4
OP4=3
OP3=2
OP2=1
OP1=0

FLOOR=0
PEAK=1
ATTACK=2
DECAY=3
SUS_LEVEL=4
RELEASE=5



'''
String FM Synth - with phase wrapping (it does not change behaviour)
PATCH NAME: STRINGS 1
OP6->OP5->OP4->OP3 |
       (R)OP2->OP1 |->out
'''

#mod_index is a vector containing the modulation indexes/amplitudes
#fr is a vector containing the frequencies ratios
#Both of these parameters are trainable
def fm_string_synth(pitch, audio_length, mod_index, fr, adsr, sampling_rate,max_ol,use_safe_cumsum=False):

    #unsqueeze adsr twice (needed for adsr library)
    adsr = torch.unsqueeze(adsr, -1)
    adsr = torch.unsqueeze(adsr, -1)

    if(use_safe_cumsum==True):
        #omega = cumsum_nd(2 * np.pi * pitch / sampling_rate, 2*np.pi)
        omega = torch.cumsum(2 * np.pi * pitch / sampling_rate, 1)
    else:
        samples = torch.ones(audio_length) #replace this with the number of samples in the input used for training
        omega = torch.cumsum(2 * np.pi * pitch * samples / sampling_rate, 1)

    # Torch unsqueeze with dim -1 adds a new dimension at the end of ol to match phases.
    #print(fr.shape) #for debug
    op6_phase =  torch.unsqueeze(fr[:, OP6], -1) * omega
    op6_output=  torch.unsqueeze(mod_index[:, OP6], -1) * torch.sin(op6_phase % (2*np.pi))
    #Apply adsr
    envelope = adsr_lib.forward(adsr[:, FLOOR+OP6], adsr[:, PEAK+OP6], adsr[:, ATTACK+OP6], adsr[:, DECAY+OP6], adsr[:, SUS_LEVEL+OP6], adsr[:, RELEASE+OP6])
    #print('envelope shape before resampling:', envelope.shape)
    envelope = adsr_lib.resample_frames(envelope, op6_output.shape[1])

    #print('synth out and envelope shape:', op6_output.shape, envelope.shape)
    op6_output = op6_output * torch.squeeze(envelope)

    
    op5_phase =  torch.unsqueeze(fr[:, OP5], -1) * omega + 2 * np.pi * op6_output
    op5_output = torch.unsqueeze(mod_index[:, OP5], -1) *torch.sin(op5_phase % (2*np.pi))
    #Apply adsr
    envelope = adsr_lib.forward(adsr[:, FLOOR+OP5], adsr[:, PEAK+OP5], adsr[:, ATTACK+OP5], adsr[:, DECAY+OP5], adsr[:, SUS_LEVEL+OP5], adsr[:, RELEASE+OP5])
    envelope = adsr_lib.resample_frames(envelope, op5_output.shape[1])

    op5_output = op5_output * torch.squeeze(envelope)

    op4_phase =  torch.unsqueeze(fr[:, OP4], -1) * omega + 2 * np.pi * op5_output
    op4_output = torch.unsqueeze(mod_index[:, OP4], -1) * torch.sin(op4_phase % (2*np.pi))
    #Apply adsr
    envelope = adsr_lib.forward(adsr[:, FLOOR+OP4], adsr[:, PEAK+OP4], adsr[:, ATTACK+OP4], adsr[:, DECAY+OP4], adsr[:, SUS_LEVEL+OP4], adsr[:, RELEASE+OP4])
    envelope = adsr_lib.resample_frames(envelope, op4_output.shape[1])

    op4_output = op4_output * torch.squeeze(envelope)

    op3_phase =  torch.unsqueeze(fr[:, OP3], -1)  * omega + 2 * np.pi * op4_output
    op3_output = torch.unsqueeze(mod_index[:, OP3], -1)  * torch.sin(op3_phase % (2*np.pi)) # output of stack of 4
    #Apply adsr
    envelope = adsr_lib.forward(adsr[:, FLOOR+OP3], adsr[:, PEAK+OP3], adsr[:, ATTACK+OP3], adsr[:, DECAY+OP3], adsr[:, SUS_LEVEL+OP3], adsr[:, RELEASE+OP3])
    envelope = adsr_lib.resample_frames(envelope, op3_output.shape[1])

    op3_output = op3_output * torch.squeeze(envelope)

    op2_phase =  torch.unsqueeze(fr[:, OP2], -1) * omega
    op2_output = torch.unsqueeze(mod_index[:, OP2], -1) * torch.sin(op2_phase % (2*np.pi))
    #Apply adsr
    envelope = adsr_lib.forward(adsr[:, FLOOR+OP2], adsr[:, PEAK+OP2], adsr[:, ATTACK+OP2], adsr[:, DECAY+OP2], adsr[:, SUS_LEVEL+OP2], adsr[:, RELEASE+OP2])
    envelope = adsr_lib.resample_frames(envelope, op2_output.shape[1])

    op2_output = op2_output * torch.squeeze(envelope)

    op1_phase =  torch.unsqueeze(fr[:, OP1], -1) * omega + 2 * np.pi * op2_output
    op1_output = torch.unsqueeze(mod_index[:, OP1], -1) * torch.sin(op1_phase % (2*np.pi)) # output of stack of 2

    #Apply adsr
    envelope = adsr_lib.forward(adsr[:, FLOOR+OP1], adsr[:, PEAK+OP1], adsr[:, ATTACK+OP1], adsr[:, DECAY+OP1], adsr[:, SUS_LEVEL+OP1], adsr[:, RELEASE+OP1])
    envelope = adsr_lib.resample_frames(envelope, op1_output.shape[1])

    op1_output = op1_output * torch.squeeze(envelope)

    return (op3_output + op1_output)/max_ol


class FMSynth(nn.Module):
    def __init__(self,sample_rate,block_size,fr, audio_length, max_ol=2,
        scale_fn = torch.sigmoid,synth_module='fmstrings'):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.audio_length = audio_length
        #self.reverb = Reverb(length=sample_rate, sample_rate=sample_rate)
        #fr = torch.tensor(fr) # Frequency Ratio
        #self.register_buffer("fr", fr) #Non learnable but sent to GPU if declared as buffers, and stored in model dictionary
        self.scale_fn = scale_fn
        self.use_cumsum_nd = False
        self.max_ol = max_ol

        available_synths = {
            'fmstrings': fm_string_synth,
        }

        self.synth_module = available_synths[synth_module]

    def forward(self,controls):

        #print("SYNTH PART-------------------") #for debug
        mod_index = self.max_ol*self.scale_fn(controls['mod_index'])
        fr = (controls['freq_ratio'])
        f0 = controls['f0']
        adsr = controls['adsr']
        #print('synth using frequencies:,', f0)
        signal = self.synth_module(f0,
                                   self.audio_length,
                                mod_index,
                                fr,
                                adsr,
                                self.sample_rate,
                                self.max_ol,
                                self.use_cumsum_nd)
        #reverb part
        #signal = self.reverb(signal)

        synth_out = {
            'synth_audio': signal,
            'mod_index': mod_index,
            'fr': fr,
            'adsr': adsr
            }
        return synth_out