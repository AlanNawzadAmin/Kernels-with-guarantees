import logging
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from torch.nn.functional import softplus
from pyro.contrib.mue.missingdatahmm import MissingDataDiscreteHMM
from pyro.contrib.mue.statearrangers import Profile
from .seq_tools import add_stops, get_lens, pad_axis

"""
This code is adapted from the pHMM model code from 
https://github.com/pyro-ppl/pyro/tree/dev/pyro/contrib/mue
used under the MIT lisence.
"""

epsilon = 1e-7
inf = 1e32
# MAX_LEN = 200
# STATE_ARRANGER = Profile(MAX_LEN)

class ProfileHMM(nn.Module):
    """
    Profile HMM.
    This model consists of a constant distribution (a delta function) over the
    regressor sequence, plus a MuE observation distribution. The priors
    are all Normal distributions, and are pushed through a softmax function
    onto the simplex.
    :param int latent_seq_length: Length of the latent regressor sequence M.
        Must be greater than or equal to 1.
    :param int alphabet_length: Length of the sequence alphabet (e.g. 20 for
        amino acids).
    :param float prior_scale: Standard deviation of the prior distribution.
    :param float indel_prior_bias: Mean of the prior distribution over the
        log probability of an indel not occurring. Higher values lead to lower
        probability of indels.
    :param float stop_prior_bias: Mean of the prior distribution over the
        log probability of a stop. Higher values lead to lower probability of
        stop until late in the sequence.
    :param float stop_prior_late: Mean of the prior distribution over the
        scaled log probability of a stop. Higher values lead to lower probability of
        stop until late in the sequence.
    :param arranger: Give a state arranger to avoid redefining one.
    :param bool stop: Include explicit stop letter.
    :param bool cuda: Transfer data onto the GPU during training.
    """
    def __init__(self, latent_seq_length, alphabet_length,
                 prior_scale=1., indel_prior_bias=10., stop_prior_bias=1.,
                 stop_prior_late=1., arranger=None,
                 stop=True, cuda=False):
        super().__init__()
        assert isinstance(cuda, bool)
        self.is_cuda = cuda

        assert latent_seq_length > 0 # and isinstance(latent_seq_length, int)
        self.latent_seq_length = latent_seq_length
        assert isinstance(alphabet_length, int) and alphabet_length > 0
        self.alphabet_length = alphabet_length

        self.precursor_seq_shape = (latent_seq_length, alphabet_length)
        self.insert_seq_shape = (latent_seq_length+1, alphabet_length)
        self.stop_shape = (2*latent_seq_length+1)
        self.indel_shape = (latent_seq_length, 3, 2)
        self.inc_stop = stop

        assert isinstance(prior_scale, float)
        self.prior_scale = prior_scale
        assert isinstance(indel_prior_bias, float)
        self.indel_prior = torch.tensor([indel_prior_bias, 0.])

        # Initialize state arranger.
        self.new_arranger = arranger is None
        if self.new_arranger is None:
            self.statearrange = Profile(self.latent_seq_length)
        else:
            self.statearrange = arranger
            self.arrange_len = arranger.M

        # Latent sequence.
        self.precursor_seq = pyro.sample("precursor_seq", dist.Normal(
                torch.zeros(self.precursor_seq_shape),
                self.prior_scale *
                torch.ones(self.precursor_seq_shape)).to_event(2))
        self.insert_seq = pyro.sample("insert_seq", dist.Normal(
                torch.zeros(self.insert_seq_shape),
                (self.prior_scale/10) *
                torch.ones(self.insert_seq_shape)).to_event(2))

        # Indel probabilities.
        self.insert = pyro.sample("insert", dist.Normal(
                self.indel_prior * torch.ones(self.indel_shape),
                self.prior_scale * torch.ones(self.indel_shape)).to_event(3))
        self.delete = pyro.sample("delete", dist.Normal(
                self.indel_prior * torch.ones(self.indel_shape),
                self.prior_scale * torch.ones(self.indel_shape)).to_event(3))
        
        # Stop probabilities
        if self.inc_stop:
            self.stop = (pyro.sample("stop", dist.Normal(
                    torch.zeros(self.stop_shape),
                    self.prior_scale * torch.ones(self.stop_shape)).to_event(1))
                + stop_prior_bias
                - stop_prior_late*torch.cat([1 - torch.arange(-latent_seq_length, 0),
                                             1 - torch.arange(-latent_seq_length-1, 0)]))
        
    def set_logits(self, normalize=True):
        if normalize:
            self.precursor_seq = self.precursor_seq - self.precursor_seq.logsumexp(-1, True)
            self.insert_seq = self.insert_seq - self.insert_seq.logsumexp(-1, True)
            self.insert = self.insert - self.insert.logsumexp(-1, True)
            self.delete = self.delete - self.delete.logsumexp(-1, True)
            # self.stop_logits = self.stop - self.stop.logsumexp(-1, True)
        if self.new_arranger:
            self.initial_logits, self.transition_logits, self.observation_logits = (
                    self.statearrange(self.precursor_seq,
                                      self.insert_seq,
                                      self.insert,
                                      self.delete,))
        else:
            self.initial_logits, self.transition_logits, self.observation_logits = (
                    self.statearrange(pad_axis(self.precursor_seq, self.arrange_len, 0, 0, use_torch=True),
                                      pad_axis(self.insert_seq, self.arrange_len + 1, 0, 0, use_torch=True),
                                      pad_axis(self.insert, self.arrange_len, 0, 0, use_torch=True),
                                      pad_axis(self.delete, self.arrange_len, 0, 0, use_torch=True)))
            keep_inds = np.r_[np.arange(self.latent_seq_length),
                              np.arange(self.latent_seq_length + 1) + self.arrange_len]
            self.initial_logits = self.initial_logits[keep_inds]
            self.transition_logits = self.transition_logits[keep_inds, :][:, keep_inds]
            self.observation_logits = self.observation_logits[keep_inds]

        if self.inc_stop:
            ninf = -1e7 * torch.ones(1)
            # Impossible to immediately stop
            self.initial_logits = torch.cat([self.initial_logits, ninf])

            # Probability of stopping from each pos
            self.transition_logits = torch.cat([
                self.transition_logits,
                self.stop[:, None]], axis=-1)
            if normalize:
                self.transition_logits = (self.transition_logits
                                          - self.transition_logits.logsumexp(-1, True))
            # Stop can only transition to stop
            stop = ninf * torch.ones([1, self.stop_shape + 1])
            stop[0, -1] = 0.
            self.transition_logits = torch.cat([self.transition_logits, stop], axis=0)

            # Only stop pos can observe stop
            self.observation_logits = torch.cat([
                self.observation_logits,
                ninf*torch.ones([1, self.insert_seq_shape[1]])], axis=0)
            stop = ninf*torch.ones([self.stop_shape + 1, 1])
            stop[-1] = 0.
            self.observation_logits = torch.cat([self.observation_logits, stop], axis=-1)
        
    def get_dist(self):
        # Construct HMM parameters.
        return MissingDataDiscreteHMM(self.initial_logits,
                                      self.transition_logits,
                                      self.observation_logits)


def observation_logits(seqs, sub_mat, normalize=False):
    """ Get a torch tensor of observation logits from sequences.
    All sequences must be of the same length and include a single stop!
    Throws an error if sequences are not of same length.
    
    Parameters:
    seqs: numpy array or torch tensor
        OHE representations of sequences all of the same length. OHE
        may possibly be padded with zeros.
    sub_mat: torch tensor
        alphabet_size X alphabet_size
    normalize: bool, default = False
        Get actual probability distribution, i.e. normalize transition
        and observation probabilities.
    
    Returns:
    len_seq: numpy array
        Length of all sequences including stop (may be smaller than OHE length). 
        Position len_seq - 1 is the stop position.
    alphabet_length: int
    observation_logits: torch tensor
        Size is shape(seqs)[:-2] + [len_seq * 2 + 1, alphabet_size + 1].
        Stops can only be seen in the stop position.
    """
    len_seq = get_lens(seqs).cpu().numpy().astype(int)
    assert np.all(len_seq == len_seq.flatten()[0]), "All seqs need to be the same length."
    len_seq = len_seq.flatten()[0]
    alphabet_size = int(seqs.shape[-1]) - 1
    # extend sub mat so stop always goes to stop
    sub_mat_stop = - inf * (1 - torch.eye(alphabet_size + 1))
    sub_mat_stop[:alphabet_size, :alphabet_size] = sub_mat
    mut_seqs = torch.matmul(seqs[..., :len_seq, :], sub_mat_stop)

    # add logits for insertions
    ins_logits = torch.zeros(list(mut_seqs.shape[:-2]) + [len_seq+1, alphabet_size+1])
    ins_logits[..., -1] = - inf
    observation_logits = torch.cat([mut_seqs, ins_logits], axis=-2)
    if normalize:
        observation_logits = (observation_logits
                              - observation_logits.logsumexp(-1, True))
    return len_seq, alphabet_size, observation_logits


def model_expand_to_obs(model, obs_logits):
    """ Take a MissingDataHMM model and replace its obs_logits.
    Also broadcasts its transition matrix and initial logits to match the
    axes of obs_logits.
    
    Parameters:
    model: MissingDataDiscreteHMM
    obs_logits: torch tensor
        Last two dimensions must match latent seq shape of model.
    
    Returns:
    model: MissingDataDiscreteHMM
    """
    model.observation_logits = obs_logits
    extra_dims = obs_logits.shape[:-2]
    
    t_shape = model.transition_logits.shape
    if len(t_shape) > 2:
        # un-broadcast the transition logits
        model.transition_logits = model.transition_logits[(0,)*(len(t_shape)-2)]
    model.transition_logits = model.transition_logits.reshape(
        (1,) * len(extra_dims) + t_shape[-2:]).expand(extra_dims + (-1, -1,))
    
    i_shape = model.initial_logits.shape
    if len(i_shape) > 1:
        # un-broadcast the initial logits
        model.initial_logits = model.initial_logits[(0,)*(len(i_shape)-1)]
    model.initial_logits = model.initial_logits.reshape(
        (1,) * len(extra_dims) + i_shape[-1:]).expand(extra_dims + (-1,))
    return model


def local_ali_phmm(seqs, sub_mat, insert_int, insert_slope, arranger=None,
                   flank_penalty=True, local_alignment=True, normalize=False):
    """ Builds a local alignment PHMM pyro distribution with seqs as latent
    sequences. This PHMM has insertions and deletions appear simultaneously:
    deletions cannot land on insertions, so it may be use to build an alignment
    kernel. Also sets last observation of the latent sequence to be a stop, which
    is an absorbing state, so all unseen emissions are a stop.
    
    Parameters:
    seqs: torch tensor
        Must include stops, but can have arbitrarily many dimensions.
    sub_mat: torch tensor
        Must be [alphabet_size, alphabet_size]. Used to derive an observation
        model from the latent sequences.
    insert_int: float
        Open gap score.
    insert_slope: float
        Extend gap score.
    arranger: Profile object
        Arranger to use for defining the pHMM.
    flank_penalty: bool, default = True
        Whether to apply any gap penalty to gaps at ends. Has no effect if
        local_alignment.
    local_alignment: bool, default = True
        If True, gaps at ends are penalized with the extension score only.
    normalize: bool, default = False
        Get actual probability distribution, i.e. normalize transition
        and observation probabilities.
        
    Returns:
    model: MissingDataDiscreteHMM
    """
    if local_alignment and not flank_penalty:
        logging.warning("flank_penalty == False ignored "
                        + "as local_alignment == True")
    len_seq, alphabet_size, obs_logits = observation_logits(seqs, sub_mat)
    
    # no penalty for deletions (counting all local alis)
    hmm = ProfileHMM(len_seq, alphabet_size+1, stop=False, cuda=True, arranger=arranger)
    # Set indel penalties
    delete_int = insert_int
    delete_slope = insert_slope
    hmm.delete *= 0
    hmm.delete[:, 0, 1] = delete_int
    hmm.delete[:, 1, 1] = delete_int
    hmm.delete[:, 2, 1] = delete_slope
    hmm.insert *= 0
    hmm.insert[:, 0, 1] = insert_int
    hmm.insert[:, 1, 1] = insert_slope
    # Deletions cannot land in insertions: insertion must be before deletion.
    hmm.insert[:, 2, 1] = - inf
    
    hmm.initial_logits = torch.zeros(len_seq)
    hmm.set_logits(normalize=normalize) # scaling doesn't matter
    model = hmm.get_dist()
    model.transition_logits = hmm.transition_logits
    
    # first set observation logits to be observed seq
    model = model_expand_to_obs(model, obs_logits)
    
    # Unobserved states are all stop: stop is absorbing
    # 1) No penalty from stop to stop
    model.transition_logits[..., len_seq-1, len_seq-1] = 0
    # 2) Can't allow for stop to insertion
    model.transition_logits[..., len_seq-1, len_seq:] = - inf
    # 3) Eliminate insertion after stop
    # (not technically necessary as stop letter must land on stop pos)
    model.transition_logits[..., :, -1] = - inf
    model.transition_logits[..., -1, :] = - inf
    if local_alignment:
        # We designate the ins before stop as the "final ins"
        # 1) No other ins can be final: no internal ins to stop
        model.transition_logits[..., len_seq:-2, len_seq-1] = - inf
        # 2) Only slope penalty to delete to stop or final ins
        model.transition_logits[..., :len_seq-1, len_seq-1] = delete_slope * torch.flip(torch.arange(len_seq-1), dims=[0])
        model.transition_logits[..., :len_seq-1, -2] = (
            insert_slope + delete_slope * torch.flip(torch.arange(len_seq-1), dims=[0]))
                
        # Deletion from first ins is linear
        model.transition_logits[..., len_seq, :len_seq-1] = delete_slope * torch.arange(len_seq-1) 
    elif not flank_penalty:
        # We designate the ins before stop as the "final ins"
        # 1) No other ins can be final: no internal ins to stop
        model.transition_logits[..., len_seq:-2, len_seq-1] = - inf
        # 2) No delete penalty to stop or final ins
        model.transition_logits[..., :len_seq-1, len_seq-1] = 0
        model.transition_logits[..., :len_seq-1, -2] = 0
        
        # No penalty for deletion from first ins
        model.transition_logits[..., len_seq, :len_seq-1] = 0
        
        # No penalty for first or last ins
        model.transition_logits[..., len_seq, len_seq] = 0
        model.transition_logits[..., -2, -2] = 0
    
    # Cannot delete and land in ins: can't start on interior ins
    model.initial_logits[..., len_seq + 1:] = - inf
    # Set insertion and deletion penalties
    if local_alignment:
        model.initial_logits[..., :len_seq] = delete_slope * torch.arange(len_seq) 
        model.initial_logits[..., len_seq] = insert_slope
        # Can start on final ins
        model.initial_logits[..., -2] = delete_slope * (len_seq - 1) + insert_slope
    elif not flank_penalty:
        # No penalty for starting with a deletion or insertion
        model.initial_logits[..., 0:len_seq+1] = 0 
        # Can start on final ins
        model.initial_logits[..., -2] = 0
    else:
        model.initial_logits[..., 0] = 0
        model.initial_logits[..., 1:len_seq] = delete_int + delete_slope * torch.arange(len_seq-1) 
        model.initial_logits[..., len_seq] = insert_int

    if normalize:
        model.initial_logits = model.initial_logits - model.initial_logits.logsumexp(-1, True)
        model.transition_logits = model.transition_logits - model.transition_logits.logsumexp(-1, True)
        model.observation_logits = model.observation_logits - model.observation_logits.logsumexp(-1, True)
    return model

