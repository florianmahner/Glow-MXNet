import mxnet as mx
import numpy as np
import math
import scipy.linalg as lg
from mxnet import nd, ndarray, gluon
from mxnet.gluon import nn

def abs_log(x):
    return nd.log(nd.abs(x))

def gaussian_log_prob(mean, log_sd, x):
    log_prob = -0.5 * (math.log(2 * math.pi) + 2. * log_sd +
                       (x - mean) ** 2 / nd.exp(2. * log_sd))

    return log_prob

def gaussian_ll(mean, log_sd, x):
    ''' Likelihood is summed probability over for each pixel of a sample. Return shape BS x Likelihood. Since we
    are in log space, it is the summed log probability of all pixels '''
    log_prob = gaussian_log_prob(mean, log_sd, x)

    return log_prob.sum(axis=(1, 2, 3))

def gaussian_sample(mean, log_sd, temp=1.0):
    ''' Sample from normal distribution at variable temperature '''
    z = nd.sample_normal(mean, log_sd.exp() * temp)

    return z

def split(x):
    ''' Split ndarray on channel dimension '''
    ch = x.shape[1]

    # If channel dimension uneven, no splitting function available in mxnet
    if ch % 2 == 1:
        ch1 = (ch // 2)  # if uneven, split_a has one dim more
        split_a = x[:, :ch1, ...]
        split_b = x[:, ch1:, ...]

    else:
        split_a, split_b = nd.split(x, axis=1, num_outputs=2)

    return split_a, split_b


class Layer(nn.HybridBlock):
    def __init(self):
        super().__init__()

    def __repr__(self):
        return repr('A layer of type ' + self.__repr__ + ' in the Glow network')

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, **kwargs):
        raise NotImplementedError

    def reverse(self, x, **kwargs):
        raise NotImplementedError


class Flow(Layer):
    ''' A step of Flow contains the following processes. Actnorm <-> Inv1x1Conv <-> Affine Coupling '''

    def __init__(self, in_channels, affine=True, filter_size=512):
        super().__init__()
        self.act_norm = ActNorm(in_channels)
        # Default is with LU decomposition
        self.inv_conv = InvConv1x1LU(in_channels)
        self.aff_coupl = AffineCoupling(in_channels, affine, filter_size)

    def forward(self, x_in):
        x_out, log_det = self.act_norm(x_in)
        x_out, det_1 = self.inv_conv(x_out)
        x_out, det_2 = self.aff_coupl(x_out)
        log_det = log_det + det_1 + det_2

        return x_out, log_det

    def reverse(self, x_out):
        x_in = self.aff_coupl.reverse(x_out)
        x_in = self.inv_conv.reverse(x_in)
        x_in = self.act_norm.reverse(x_in)

        return x_in


class ActNorm(Layer):
    def __init__(self, in_channels):
        super().__init__()

        # Scale = std, Bias = mean
        with self.name_scope():
            self.scale = self.params.get('scale', shape=(1, in_channels, 1, 1), init=mx.init.Zero())
            self.bias = self.params.get('bias,', shape=(1, in_channels, 1, 1), init=mx.init.Zero())

        self.scale.initialize()
        self.bias.initialize()

        self.init_flag = False

    def _init(self, x_in):
        x = x_in.transpose((1, 0, 2, 3))
        x = x.flatten()

        bias = x.mean(axis=1).expand_dims(1)
        scale = (x - bias).square().mean(axis=1).sqrt()

        # Post-norm activations after first forward call have zero mean and unit variance. We therefore initialize
        # with the negative bias (mean) and inverse scale (std).
        bias = bias * -1
        scale = 1 / (scale + 1e-6)
        
        with mx.autograd.pause():
            self.scale.set_data(scale.reshape(1, scale.size, 1, 1))
            self.bias.set_data(bias.reshape(1, bias.size, 1, 1))

        self.init_flag = True

    def forward(self, x_in):
        _, _, h, w = x_in.shape
        ctx = x_in.context

        # Initialize scale and variance in first forward pass
        if not self.init_flag:
            self._init(x_in)

        # Get log-determinant of the scale
        log_abs = abs_log(self.scale.data(ctx))
        log_det = h * w * log_abs.sum()

        x_out = self.scale.data(ctx) * (x_in + self.bias.data(ctx))

        return x_out, log_det

    def reverse(self, x_out):
        ctx = x_out.context
        return (x_out / self.scale.data(ctx)) - self.bias.data(ctx)


class InvConv1x1(Layer):
    ''' Replaces fixed permutation through 1x1 convolution that inherently does random permutation over channels '''

    def __init__(self, in_channels):
        super().__init__()
        weights = np.random.randn(in_channels, in_channels)
        q, _ = lg.qr(weights)
        weights = nd.array(q.reshape(*q.shape, 1, 1))

        with self.name_scope():
            self.weights = self.params.get('weights', shape=(in_channels, in_channels, 1, 1))

        self.weights.initialize()
        self.weights.set_data(weights)

    def forward(self, x_in):
        _, c, h, w = x_in.shape
        ctx = x_in.context

        # Kernel for h and w is 1, since 1x1 convolution
        x_out = nd.Convolution(x_in, self.weights.data(ctx), kernel=(1, 1), no_bias=True, num_filter=c)

        # Computes the signed log determinant of the weight matrix -> h * w * log(|det(W)|)
        log_det = h * w * ndarray.linalg_slogdet(self.weights.data(ctx).squeeze())[1]
        #log_det = h * w * nd.log(abs(nd.linalg.det(self.weights.data(ctx).squeeze())))

        return x_out, log_det

    def reverse(self, x_out):
        _, c, h, w = x_out.shape
        ctx = x_out.context

        weights = nd.linalg.inverse(self.weights.data(ctx).squeeze())
        weights = weights.reshape(*weights.shape, 1, 1)

        x_in = nd.Convolution(x_out, weights, kernel=(1, 1), no_bias=True, num_filter=c)

        return x_in


class InvConv1x1LU(Layer):
    ''' Naive way has comp. complexity of O(c^3) -> With LU Decomposition can be reduced to O(c). '''

    def __init__(self, n_channels):
        super().__init__()

        weights = np.random.randn(n_channels, n_channels).astype('float32')
        q, _ = lg.qr(weights)
        w_p, w_l, w_u = lg.lu(q)
        w_s = np.diag(w_u)
        log_w_s = np.log(abs(w_s))
        w_u = np.triu(w_u, k=1)

        # Register value for L, U and s -> To be optimized. Permutation matrix w_p remains fixed
        # Ensures that parameters in the comp. graph are all in the samecontext, either trainable or not
        with self.name_scope():
            self.w_l = self.params.get('w_l', shape=(n_channels, n_channels))
            self.w_s = self.params.get('w_s', shape=(n_channels))
            self.w_u = self.params.get('w_u', shape=(n_channels, n_channels))
            self.w_p = self.params.get('w_p', shape=(n_channels, n_channels), differentiable=False)

        for name, val in zip(['w_l', 'w_s', 'w_u', 'w_p'], [w_l, log_w_s, w_u, w_p]):
            param = self.params.get(name)
            param.initialize()
            param.set_data(nd.array(val))

        self.u_mask = self.params.get('u_mask', shape=(n_channels, n_channels), differentiable=False)
        self.l_mask = self.params.get('l_mask', shape=(n_channels, n_channels), differentiable=False)
        self.s_sign = self.params.get('s_sign', shape=(n_channels), differentiable=False)
        self.l_eye = self.params.get('l_eye', shape=(n_channels, n_channels), differentiable=False)

        self.u_mask.initialize()
        self.u_mask.set_data(nd.array(np.triu(np.ones_like(w_u), k=1)))
        self.l_mask.initialize()
        self.l_mask.set_data(self.u_mask.data().transpose())
        self.s_sign.initialize()
        self.s_sign.set_data(nd.array(np.sign(w_s)))
        self.l_eye.initialize()
        self.l_eye.set_data(nd.array(np.eye(self.l_mask.shape[0])))

    def get_weights(self, ctx):
        weights = nd.dot(self.w_p.data(ctx), 
                  nd.dot((self.w_l.data(ctx) * self.l_mask.data(ctx) + self.l_eye.data(ctx)), 
                        ((self.w_u.data(ctx) * self.u_mask.data(ctx)) + nd.diag(self.s_sign.data(ctx) * 
                        nd.exp(self.w_s.data(ctx))))))

        weights = weights.reshape(0, 0, 1, 1) # expand dims

        return weights

    def forward(self, x_in):
        _, c, h, w = x_in.shape
        ctx = x_in.context
        weights = self.get_weights(ctx)
        x_out = nd.Convolution(x_in, weights, kernel=(1, 1), no_bias=True, num_filter=c)
        log_det = h * w * self.w_s.data(ctx).sum()
        
        return x_out, log_det

    def reverse(self, x_out):
        c = x_out.shape[1]
        weights = self.get_weights(x_out.context).squeeze()
        weights = nd.linalg.inverse(weights)
        weights = weights.reshape(0, 0, 1, 1)
        x_in = nd.Convolution(x_out, weights, kernel=(1, 1), no_bias=True, num_filter=c)

        return x_in


class ZeroConv2D(Layer):
    ''' Initialize last layer of each NN in affine coupling with identity function in first forward pass '''

    def __init__(self, in_channels, out_channels, log_scale_factor=3.):
        super().__init__()

        # Weights are learned. Initial forward pass is the identity function
        with self.name_scope():
            self.conv = nn.Conv2D(out_channels, in_channels=in_channels, kernel_size=3,
                                  weight_initializer='zeros', bias_initializer='zeros')
            self.log_s = self.params.get('log_s', shape=(1, out_channels, 1, 1), init=mx.init.Zero())

        self.conv.initialize()
        self.log_s.initialize()
        self.log_scale_factor = log_scale_factor

    def forward(self, x_in):
        # Pad along height and width and learn the identity function
        x_out = nd.pad(x_in, mode='constant', pad_width=(0, 0, 0, 0, 1, 1, 1, 1), constant_value=1)
        x_out = self.conv(x_out)
        # Not in paper, but in the glow code
        x_out = x_out * nd.exp(self.log_s.data(x_in.context) * self.log_scale_factor)



        return x_out


class AffineCoupling(Layer):
    ''' Affine Coupling block. First splits the input, then forward pass through the network to get scale and shift 
    operation for one split. Concatenates with other split (original input) to combine into layer output '''

    def __init__(self, in_channels, affine=True, filter_size=512):
        super().__init__()

        self.affine = affine

        self.net = nn.HybridSequential()
        with self.name_scope():
            self.net.add(nn.Conv2D(filter_size, kernel_size=3, padding=1, in_channels=in_channels//2))
            self.net.add(nn.Activation('relu'))
            self.net.add(nn.Conv2D(filter_size, kernel_size=1))
            self.net.add(nn.Activation('relu'))

            # Num output channels depends on if the coupling is additive or affine. Additive does not concatenate back.
            if self.affine:
                self.net.add(ZeroConv2D(filter_size, in_channels))
            else:
                self.net.add(ZeroConv2D(in_channels, in_channels//2))

        # Biases are zero and weights sampled from normal distribution
        self.net[0].weight.initialize(init=mx.init.Normal(sigma=0.05), force_reinit=True)
        self.net[0].bias.initialize(init=mx.init.Zero(), force_reinit=True)
        self.net[2].weight.initialize(init=mx.init.Normal(sigma=0.05), force_reinit=True)
        self.net[2].bias.initialize(init=mx.init.Zero(), force_reinit=True)

    def forward(self, x_in):
        # in_b has one channel less if the number of channels is uneven
        z_1, z_2 = split(x_in)

        if self.affine:
            h = self.net(z_1)
            scale, shift = split(h)  # shift has two channels
            scale = nd.sigmoid(scale + 2.0)
            z_2 = z_2 + shift
            z_2 = z_2 * scale
            log_det = scale.log().flatten()
            log_det = log_det.sum()

        # If affine is false the coupling is simply additive. Has zero log determinant
        else:
            z_2 = z_2 + self.net(z_1)
            log_det = 0

        x_out = nd.concat(z_1, z_2, dim=1)

        return x_out, log_det

    def reverse(self, x_out):
        # Split - Forward - Concat
        z_1, z_2 = split(x_out)

        if self.affine:
            h = self.net(z_1)
            scale, shift = split(h)
            scale = nd.sigmoid(scale + 2.0)
            z_2 = z_2 / scale
            z_2 = z_2 - shift

        else:
            z_2 = z_2 - self.net(z_1)

        x_in = nd.concat(z_1, z_2, dim=1)

        return x_in


class Squeeze2D(Layer):
    ''' Reshape along height and width. First divide image into subsquares of shape 2x2xc and 
    then concat on channel dimension to shape 1x1x4c '''

    def __init__(self):
        super().__init__()

    def forward(self, x_in):
        log_det = 0  # squeezing does not change the log determinant
        bs, c, h, w = x_in.shape
        assert h % 2 == 0 and w % 2 == 0, 'Uneven height and width to further squeeze. L probably too large.'

        x_out = x_in.reshape((bs, c, h//2, 2, w//2, 2))
        x_out = x_out.transpose((0, 1, 3, 5, 2, 4))
        x_out = x_out.reshape((bs, c*4, h//2, w//2))

        return x_out, log_det

    def reverse(self, x_out):
        # Unsqueeze the input again
        bs, c, h, w = x_out.shape
        assert c >= 4 and c % 4 == 0, 'Number of channels is not a factor of 4'

        x_in = x_out.reshape((bs, c//4, 2, 2, h, w))
        x_in = x_in.transpose((0, 1, 4, 2, 5, 3))
        x_in = x_in.reshape((bs, c//4, h*2, w*2))

        return x_in


class Split2D(Layer):
    ''' Defines the Splitting operation after a block of K Flow steps.'''

    def __init__(self, in_channels):
        super().__init__()
        # Double num. of channels to sample means and sd per pixel equally per channel
        self.conv = ZeroConv2D(in_channels, in_channels * 2)

    def forward(self, x_in):
        # Factors out half of the dimensions - Gaussianize other half
        z_1, z_2 = split(x_in)
        # sample mean and std from first half of the input
        mean, log_sd = split(self.conv(z_1))
        # get log prob p. pixel in other half given sampled mean+std
        log_det = gaussian_ll(mean, log_sd, z_2)

        return z_1, log_det  # NOTE z_1 is afterwards also squeezed in tf-openai code

    def reverse(self, x_out, temperature=1.0):
        z_1 = x_out
        mean, log_sd = split(self.conv(z_1))
        z_2 = gaussian_sample(mean, log_sd, temperature)
        z = nd.concat(z_1, z_2)

        return z
