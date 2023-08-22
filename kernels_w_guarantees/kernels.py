import logging
import numpy as np
import torch
from .profile_hmm import local_ali_phmm, observation_logits
from pyro.contrib.mue.statearrangers import Profile
from .seq_tools import (add_stops, get_lens, get_ohe,
                        pad_axis, hamming_dist, pad_seq_len, pad_seq_len_torch)

epsilon = 1e-30
def map_in_batch(f, x, batch_size, axis_keep):
    """ Map f onto x in batches.
    
    Parameters:
    f: function
    x: numpy array or torch tensor
    batch_size: int
    axis_keep: int
        x can have arbitrarily many axes. f is mapped onto a flattened
        x of shape [batch_size] + x.shape[-axis_keep:].
        
    Returns:
    out: numpy array
        Of shape x.shape[:-axis_keep].
    """
    shape = x.shape
    x_flat = x.reshape((-1,) + shape[-axis_keep:])
    n = len(x_flat)
    f_out_dim = np.shape(f(x_flat[[0]]))[1:]
    out = np.empty((n,) + f_out_dim)
    for i in range(int(np.ceil(n / batch_size))):
        x_batch = x_flat[i*batch_size : (i+1)*batch_size]
        out[i*batch_size : (i+1)*batch_size] = f(x_batch)
    return out.reshape(shape[:-axis_keep] + f_out_dim)

############ Alignment kernel ##################


def get_ali_kernel(sub_mat, open_gap_score, extend_gap_score,
                   alphabet_size, flank_penalty=True, local_alignment=True,
                   w_stop=False, max_len=10,
                   batch_size=150, normalize=False, sub_beta=1, dtype=np.float64):
    """ Create an alignment kernel.
    
    Parameters:
    sub_mat: float
        substitution matrix
    open_gap_score: float
        - mu
    extend_gap_score: float
        - (mu + Delta mu)
    alphabet_size: int
    flank_penalty: bool, default = False
        If False, get a local alignment kernel
        - no gap scores at the beginning or end.
    local_alignment: bool, default = True
        If True, do not apply open_gap_score to gaps at beginning or end
        - get "no ends" alignment kernel.
    w_stop: bool, default = True
        If True, the kernel only accepts sequences with a single stop and
        if False, the kernel accepts only sequences without a stop row.
    max_len: int, default = 10
        Maximum length of sequence (including stop) accepted by kernel.
    batch_size: int, default = 150
        Number of sequences compared at once during calculations.
    normalize: bool, default = False
        If True, normalize kernel.
    sub_beta: float, default = 1
        Score for matching letters if sub_mat is not given.
    
    Returns:
    kern_func: function
        The kernel. Takes OHE sequences, numpy arrays without stops if not w_stop
        and arrays or tensors with stops if w_stop. Can accept sequences of arbitrarily
        many dimensions. kern_func(seqs) returns the Gram matrix of seqs and 
        kern_func(seqs1, seqs2) compares seqs1 and seqs2; in the later case, both OHEs
        must be the same length, and one should set seqs_x to be the shorter list.
    """
    
    # Set up substitution matrix
    if sub_mat is None:
        sub_mat = sub_beta * torch.eye(alphabet_size, dtype=torch.float64)
    sub_mat = torch.nan_to_num(torch.log(sub_mat), neginf=-1e32)

    # Initialize phmms for each length with dummy latent sequences
    arranger = Profile(max_len)
    phmms = [local_ali_phmm(torch.tensor(np.tile(np.eye(alphabet_size+1, dtype=dtype)[[-1]], (i+1, 1))),
                            sub_mat, open_gap_score, extend_gap_score,
                            local_alignment=local_alignment,
                            flank_penalty=flank_penalty, arranger=arranger)
             for i in range(max_len)]

    def kern_func(seqs1_w_stop, seqs2_w_stop=None):
        # Get tensor sequences with stops.
        if w_stop:
            seqs1 = torch.tensor(seqs1_w_stop)
        else:
            seqs1 = add_stops(seqs1_w_stop)
        if seqs2_w_stop is None:
            two_is_one = True
            seqs2_w_stop = seqs1_w_stop
            seqs2 = seqs1
        else:
            two_is_one = False
            if w_stop:
                seqs2 = torch.tensor(seqs2_w_stop)
            else:
                seqs2 = add_stops(seqs2_w_stop, dtype=dtype)

        shape1 = seqs1.shape[:-2]
        shape2 = seqs2.shape[:-2]
        lens = np.max([np.shape(seqs1)[-2], np.shape(seqs2)[-2]])
        seqs1 = pad_seq_len_torch(seqs1.reshape((-1,) + seqs1.shape[-2:]), lens)
        seqs2 = pad_seq_len_torch(seqs2.reshape((-1,) + seqs2.shape[-2:]), lens)
        seq_shape = seqs1.shape[-2:]        
        ker_mat = np.zeros([np.prod(shape1), np.prod(shape2)])
        log_diag_x = np.zeros(np.prod(shape1))
        log_diag_y = np.zeros(np.prod(shape2))
        # Cycle through seqs1 and apply to seqs2.
        for i, seq1 in enumerate(seqs1):
            if torch.any(torch.isnan(seq1)):
                ker_mat[i] = -np.inf
                log_diag_x[i] = np.inf
            else:
                # Set up PHMM.
                seq1_hmm = phmms[int(torch.sum(seq1))-1]
                seq1_hmm.observation_logits = observation_logits(seq1, sub_mat)[-1]
                # Set up query sequences and get liks.
                if not two_is_one:
                    seqs_b = seqs2
                    liks = map_in_batch(lambda seqs: seq1_hmm.log_prob(seqs).cpu().numpy(),
                                        seqs2, batch_size, 2)
                    ker_mat[i] = liks
                    log_diag_x[i] = seq1_hmm.log_prob(seq1).cpu().numpy()
                else:
                    # Compare seq i with averything after i.
                    seqs_b = seqs2[i:]
                    liks = map_in_batch(lambda seqs: seq1_hmm.log_prob(seqs).cpu().numpy(),
                                        seqs2[i:], batch_size, 2)
                    ker_mat[i, i:] = liks
                    ker_mat[i:, i] = ker_mat[i, i:]
                    log_diag_x[i] = liks[0]
                    log_diag_y[i] = log_diag_x[i]

        # Get log_diag_y if seqs2 is not seqs1.
        if not two_is_one and normalize:
            for i, seq2 in enumerate(seqs2):
                if torch.any(torch.isnan(seq1)):
                    log_diag_y[i] = np.inf
                else:
                    # set up phmm
                    seq2_hmm = phmms[int(torch.sum(seq2))-1]
                    seq2_hmm.observation_logits = observation_logits(seq2, sub_mat)[-1]
                    # get diag lik
                    log_diag_y[i] = seq2_hmm.log_prob(seq2).cpu().numpy()
        
        # Exponentiate.
        if not normalize:
            ker_mat = np.exp(ker_mat)
        else:
            ker_mat = np.exp(ker_mat - 0.5*log_diag_x[:, None] - 0.5*log_diag_y[None, :])
        return ker_mat.reshape(shape1 + shape2)       
    return kern_func


########## k-mer spectrum kernel ##############


def seq_filter(seqs, filters):
    """ Applies filters to OHE seqs.
    
    Parameters:
    seqs: numpy array
        Must be OHE seqs of any number of dimensions.
    filters: numpy array
        Must be set of filters of any number of dimensions.
        Last dimension must be alphabet_len, and must match the last
        dim of seqs.
        
    Returns:
    conv: numpy array
        Shape np.shape(seqs)[:-2] + np.shape(filters)[:-2]
        + [len of seq]. End of len is convolution with padded 0s.
    """
    # first pad seqs and filters
    seqs_len = np.shape(seqs)[-2]
    filter_len = np.shape(filters)[-2]
    pad_to_len = seqs_len + filter_len
    pad_seqs = pad_seq_len(seqs, pad_to_len)
    pad_filters = pad_seq_len(filters, pad_to_len)
    # first reverse the filters so that i, j -> -i, -j (including 0!)
    rev_filters = pad_filters[..., ::-1, ::-1]
    rev_filters = np.concatenate([rev_filters[..., [-1], :],
                                  rev_filters[..., :-1, :]], axis=-2)
    rev_filters = np.concatenate([rev_filters[..., :, [-1]],
                                  rev_filters[..., :, :-1]], axis=-1)
    # now convolve
    fl = ''.join(['a', 'b', 'c', 'd', 'e'][:len(np.shape(filters))-2])
    conv = np.fft.ifft2(np.einsum('...ij,{}ij->...{}ij'.format(fl, fl),
                                  np.fft.fft2(pad_seqs),
                                  np.fft.fft2(rev_filters)))
    # drop cycling alphabet and cyling len
    conv = conv[..., :-filter_len, 0]
    return conv


def get_ohe_of_all_kmers(k, alphabet_size):
    """ Get a list of all k-mers OHE.
    
    Parameters:
    k: int
    alphabet_size: int
    
    Returns:
    kmers: numpy array
        Shape is [num_kmers, k, alphabet_size] where num_kmers = 
        alphabet_size * k.
    """
    kmers = np.zeros(k*[alphabet_size] + [k, alphabet_size])
    for pos in range(k):
        for b in range(alphabet_size):
            kmers[(slice(None),)*pos + (b,) 
                  + (slice(None),)*(k-pos-1) + (pos, b,)] = 1
    return kmers.reshape([-1, k, alphabet_size])


def fast_kmer_counter(seqs, k=3, batch_size=40):
    """ Counts all kmers in seqs by convolving k-mer OHE filters
    with the OHE seqs. Quite fast: about 10^5 letters in
    2 seconds for k=3.
    
    Parameters:
    seqs: numpy array
        OHE seqs of any number of dimensions.
    k: int
    
    Returns:
    kmer_counts: numpy array
        Shape is np.shape(seqs)[:-2] + [num_kmers].
    """
    len_seqs, alphabet_size = np.shape(seqs)[-2:]
    kmers = get_ohe_of_all_kmers(k, alphabet_size)
    kmers = pad_axis(kmers, len_seqs, -2, 0)
    #reverse the kmer so that convolution is filtering
    inds = map_in_batch(lambda x: seq_filter(x, kmers),
                        seqs, batch_size, 2)
    return np.sum(np.isclose(inds, k), axis=-1)


def kmer_kernel(seqs_x, seqs_y=None, k=3, normalize=False):
    """ kernel which uses k-mer counts as features.
    
    Parameters:
    seqs_x: numpy array
        OHE seqs of any length.
    seqs_y: numpy array, default=None
        OHE seqs of any length. If None, uses seqs_x.
    k: int
        length of k-mers defining features.
    normalize: bool
        Whether to normalize the kernel.

    Returns:
    mat: numpy array
        Kernel values of shape
        np.shape(seqs_x)[:-2] + np.shape(seqs_y)[:-2]
    """    
    # kmer counter takes nans to 0
    counts_x = fast_kmer_counter(seqs_x)
    if seqs_y is not None:
        counts_y = fast_kmer_counter(seqs_y)
    else:
        counts_y = counts_x
    if normalize:
        counts_x = counts_x / np.sqrt(np.sum(counts_x**2, axis=-1))[..., None]
        counts_y = counts_y / np.sqrt(np.sum(counts_y**2, axis=-1))[..., None]
    return np.tensordot(counts_x, counts_y, axes=(-1, -1))

############ Hamming kernel ##################


def hamming_ker_exp(seqs_x, seqs_y=None, alphabet_name='dna', bandwidth=1, lag=1):
    if seqs_y is None:
        seqs_y = seqs_x
    h_dists = hamming_dist(seqs_x, seqs_y, alphabet_name=alphabet_name, lag=lag)
    return np.exp(- h_dists/bandwidth)


def hamming_ker_dot(seqs_x, seqs_y=None, alphabet_name='dna', lag=1):
    if seqs_y is None:
        seqs_y = seqs_x
    h_dists = hamming_dist(seqs_x, seqs_y, alphabet_name=alphabet_name, lag=lag)
    x_lens = get_lens(get_ohe(seqs_x))
    y_lens = get_lens(get_ohe(seqs_y))
    max_len = np.max([np.tile(x_lens[:, None], (1, len(y_lens))),
                      np.tile(y_lens[None, :], (len(x_lens), 1))], axis=0)
    dot = max_len - h_dists
    return dot / np.sqrt(x_lens[:, None] * y_lens[None, :])


def hamming_ker_imq(seqs_x, seqs_y=None, alphabet_name='dna', scale=1, beta=1/2, lag=1):
    if seqs_y is None:
        seqs_y = seqs_x
    h_dists = hamming_dist(seqs_x, seqs_y, alphabet_name=alphabet_name, lag=lag)
    return (1+scale) ** beta / (scale + h_dists) ** beta


############ Building vector field kernels #############


def get_len_ker(kernel):
    """ Take a scalar field kernel and build a kernel such that
    k((X, Y), (X', Y')) is 0 if |X|=|Y| or |X'|=|Y'|, and equal to
    k(X, X') if |X|<|Y| and |X'|<|Y'|.
    
    Parameters:
    kernel: function
        Must be able to take a single numpy 3-D OHE set of sequences
        and return a Gram metrix, as well as two sets of sequences
        and return their comparisons.
        
    Returns:
    len_ker: function
        Takes 3-D xs, 4-D ys, and returns a Gram matrix as a vector
        field kernel.
    """
    def len_ker(xs, ys):
        seq_lens_x = get_lens(xs)
        seq_lens_y = get_lens(ys)
        num_seqs, num_muts = np.shape(ys)[:-2]
        
        dels = seq_lens_y < seq_lens_x[:, None]
        ins = seq_lens_y > seq_lens_x[:, None]
        ys_del = ys[dels, :, :]
            
        vf_ker_mat = np.zeros([num_seqs, num_muts, num_seqs, num_muts])
        xs_mat = kernel(xs)
        # Those edges that are insertions are just x-x comparisons, and +ve
        vf_ker_mat += (xs_mat[:, None, :, None]
                       * ins[None, None, :, :] * ins[:, :, None, None])
        if np.sum(dels)>0:
            # Those edges that are insertions-to-deletions are x-to-y and -ve.
            # Don't calculate x-to-y for all ys, just those that are deletions
            xy_mat_del = kernel(xs, ys_del)
            xy_mat = np.zeros([num_seqs, num_seqs, num_muts])
            xy_mat[:, dels] = xy_mat_del
            vf_ker_mat += - (xy_mat[:, None, :, :]
                             * dels[None, None, :, :] * ins[:, :, None, None])
            vf_ker_mat += - (np.transpose(xy_mat, (1, 2, 0))[:, :, :, None]
                             * ins[None, None, :, :] * dels[:, :, None, None])
            
            # Those edges that are deletions-to-deletions are y-to-y and +ve.
            yy_mat_del = kernel(ys_del, ys_del)
            yy_mat = np.zeros([num_seqs, num_muts, num_seqs, num_muts])
            cross_dels = (dels[None, None, :, :]
                          * dels[:, :, None, None]).astype(bool)
            yy_mat[cross_dels] = yy_mat_del.flatten()
            vf_ker_mat += (yy_mat
                               * dels[None, None, :, :] * dels[:, :, None, None])
        return vf_ker_mat
    return len_ker


def get_sq_ker(kernel):
    """ Take a scalar field kernel and build a kernel
    (k(X, X') + k(Y, Y'))^2 if sign(X, Y) = 1 and sign(X', Y') = 1.
    
    Parameters:
    kernel: function
        Must be able to take a single numpy 3-D OHE set of sequences
        and return a Gram metrix, as well as two sets of sequences
        and return their comparisons.
        
    Returns:
    sq_ker: function
        Takes 3-D xs, 4-D ys, and signs, and returns a Gram matrix
        as a vector field kernel. ohe shapes must be the same.
    """
    def sq_ker(xs, ys, signs):
        seq_lens_x = get_lens(xs)
        seq_lens_y = get_lens(ys)
        num_seqs, num_muts = np.shape(ys)[:-2]
        
        broadcast_x = np.tile(xs[:, None], (1, num_muts, 1, 1))
        # l_seqs are X if s = 1 and Y otherwise, u_seqs are opposite
        l_seqs = np.copy(ys)
        u_seqs = np.copy(ys)
        l_seqs[signs == 1] = broadcast_x[signs == 1]
        u_seqs[signs == -1] = broadcast_x[signs == -1]
        vf_ker_mat = (kernel(l_seqs) + kernel(u_seqs))**2
        vf_ker_mat *= signs[None, None, :, :] * signs[:, :, None, None]
        return vf_ker_mat
    return sq_ker


def vs_kern(coerce_ker, delta_ker, sign_ker=None):
    """ Takes two kernels to create a coercive and deltable
    vector space kernel. Does so by summing a len_kernel and a sq_kernel.
    
    Parameters:
    coerce_ker: function
        Scalar field kernel to build len_kernel.
    delta_ker: function
        Scalar field kernel to build sq_kernel.
    sign_ker: function, default = None
        Optional kernelto use k^\nabla to determine signs between
        sequences of the same len.
    
    Returns:
    vf_ker: function
        Takes 3-D xs, 4-D ys, and signs, and returns a Gram matrix
        as a vector field kernel. ohe shapes must be the same.
    """
    ker_1 = get_len_ker(coerce_ker)
    ker_2 = get_sq_ker(delta_ker)
    def vf_ker(xs, ys, signs):    
        if sign_ker is not None:
            base_seq_x = xs[0] * 0
            base_seq_y = np.copy(xs[0])
            base_seq_y[1:] = 0
            sign_ker_mat = (sign_ker(base_seq_x[None], xs)[0, :, None]
                            + sign_ker(base_seq_y[None], ys)[0]
                            - sign_ker(base_seq_x[None], ys)[0]
                            - sign_ker(base_seq_y[None], xs)[0, :, None])
            assert np.all(sign_ker_mat != 0)
            signs = sign_ker_mat > 0
        return ker_1(xs, ys) + ker_2(xs, ys, signs)
    return vf_ker
