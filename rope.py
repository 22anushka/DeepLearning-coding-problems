import numpy as np
# can similarly be done in torch

def np_polar(r, theta):
  return r * np.exp(1j * theta)


def precompute_theta_pos_frequency(head_dim, seq_len, device, theta: float = 10000.0):
  # theta is as given in the paper
  assert head_dim % 2 == 0, "Dimension must be divisible by 2"
  powers = np.arrange(0, head_dim, 2) # so that we get 0, 2, 4, .... which mimics i / d/2 or 2i / d
  denominator = theta ** powers/head_dim

  # the position parameters
  m = np.arrange(seq_len)

  # now, every position needs to be multiplied by every theta.
  # so pos1 by column 1 containing mtheta1, mtheta2,... and similarly for the other positions and columns

  # to do this, basically, outper product
  # shape = (seq_len) outer (seq_len, dim/2) -> (seq_len, dim/2)
  freqs = np.outer(m, theta)

  # compute complex numbers in the polar form 
  # we are doing this because now we are dealing with relative position and hence representing the angles in the polar coordinate
  # Polar coordinate -> Re^(i*theta) = Rcos(theta) + i*Rsin(theta)
  # we let R = 1
  # we are just converting the frequencies by taking their cos and sign values. Still (seq_len, dim/2)
  freq_complex = np_polar(np.ones_like(freqs), freqs) # where the second argument is theta, first argument is R
  # with torch, you can just use torch.polar
  return freq_complex


def apply_rotary(x, freq_complex):
  # group two consecutive dimensions (according to the formula)
  # x = (b, seq_len, n, head_dim) -> (b, seq_len, n, head_dim/2)
  # if using torch
  # x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2)) # want to group 2

  # Reshape to (..., -1, 2), grouping values in pairs
  # the * before x is used to represent "unpacking" the tuple holding the shape -> as arguments
  x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
  x_complex = x_reshaped[..., 0] + 1j * x_reshaped[..., 1]

  # (seq_len, head_dim/2) -> (1, seq_len,  1,  head_dim/2)
  freq_complex = np.expand_dims(freq_complex, axis=(0, 2)) # same as unsqueeze(0).unsqueeze(2)

  # multiply
  # (b, seq_len, n, head_dim/2) * (1, seq_len, 1, head_dim/2)
  x_rotated = x_complex * freq_complex

  # x_out = torch.view_as_real(x_rotated) # in torch
  x_out = np.stack([x_rotated.real, x_rotated.imag], axis=-1)

  # have to reshape back to flatten it out
  x_out = x.reshape(*x.shape)
  return x_out



