import numpy as np
import torch
import torch.nn as nn

## Adapted from diffsynth project

def soft_clamp_min(x, min_v, T=100):
    return torch.sigmoid((min_v-x)*T)*(min_v-x)+x

def resample_frames(inputs, n_timesteps, mode='linear', add_endpoint=True):
    """interpolate signals with a value each frame into signal with a value each timestep
    [n_frames] -> [n_timesteps]

    Args:
        inputs (torch.Tensor): [n_frames], [batch_size, n_frames], [batch_size, n_frames, channels]
        n_timesteps (int): 
        mode (str): 'window' for interpolating with overlapping windows
        add_endpoint ([type]): I think its for windowed interpolation
    Returns:
        torch.Tensor
        [n_timesteps], [batch_size, n_timesteps], or [batch_size, n_timesteps, channels?]
    """
    orig_shape = inputs.shape
    
    if len(orig_shape)==1:
        inputs = inputs.unsqueeze(0) # [dummy_batch, n_frames]
        inputs = inputs.unsqueeze(1) # [dummy_batch, dummy_channel, n_frames]
    if len(orig_shape)==2:
        inputs = inputs.unsqueeze(1) # [batch, dummy_channel, n_frames]
    if len(orig_shape)==3:
        inputs = inputs.permute(0, 2, 1) # # [batch, channels, n_frames]

    if mode == 'window':
        raise NotImplementedError
        # upsample_with_windows(outputs, n_timesteps, add_endpoint)
    else:
        # interpolate expects [batch_size, channel, (depth, height,) width]
        outputs = nn.functional.interpolate(inputs, size=n_timesteps, mode=mode, align_corners=not add_endpoint)
    
    if len(orig_shape) == 1:
        outputs = outputs.squeeze(1) # get rid of dummy channel 
        outputs = outputs.squeeze(0) #[n_timesteps]
    if len(orig_shape) == 2:
        outputs = outputs.squeeze(1) # get rid of dummy channel # [n_frames, n_timesteps] 
    if len(orig_shape)==3:
        outputs = outputs.permute(0, 2, 1) # [batch, n_frames, channels]

    return outputs


def forward(floor, peak, attack, decay, sus_level, release, noise_mag=0.0, note_off=0.8, n_frames=250, min_value=0.0, max_value=1.0,
           channels=1):
    """generate envelopes from parameters
    Args:
        floor (torch.Tensor): floor level of the signal 0~1, 0=min_value (batch, 1, channels)
        peak (torch.Tensor): peak level of the signal 0~1, 1=max_value (batch, 1, channels)
        attack (torch.Tensor): relative attack point 0~1 (batch, 1, channels)
        decay (torch.Tensor): actual decay point is attack+decay (batch, 1, channels)
        sus_level (torch.Tensor): sustain level 0~1 (batch, 1, channels)
        release (torch.Tensor): release point is attack+decay+release (batch, 1, channels)
        note_off (float or torch.Tensor, optional): note off position. Defaults to 0.8.
        n_frames (int, optional): number of frames. Defaults to None.
    Returns:
        torch.Tensor: envelope signal (batch_size, n_frames, 1)
    """

    #print('attack shape:', attack.shape)
    
    torch.clamp(floor, min=0, max=1)
    torch.clamp(peak, min=0, max=1)
    torch.clamp(attack, min=0, max=1)
    torch.clamp(decay, min=0, max=1)
    torch.clamp(sus_level, min=0, max=1)
    torch.clamp(release, min=0, max=1)
    batch_size = attack.shape[0]
    # batch, n_frames, 1
    x = torch.linspace(0, 1.0, n_frames)[None, :, None].repeat(batch_size, 1, channels)
    x = x.to(attack.device)
    attack = attack * note_off
    A = x / (attack)
    A = torch.clamp(A, max=1.0)
    D = (x - attack) * (sus_level - 1) / (decay+1e-5)
    D = torch.clamp(D, max=0.0)
    D = soft_clamp_min(D, sus_level-1)
    S = (x - note_off) * (-sus_level / (release+1e-5))
    S = torch.clamp(S, max=0.0)
    S = soft_clamp_min(S, -sus_level)
    peak = peak * max_value + (1 - peak) * min_value
    floor = floor * max_value + (1 - floor) * min_value
    signal = (A + D + S + torch.randn_like(A)*noise_mag)*(peak - floor) + floor
    return torch.clamp(signal, min=min_value, max=max_value)