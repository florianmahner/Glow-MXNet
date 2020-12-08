import mxnet as mx
import numpy as np
import math
from mxnet import nd, ndarray, gluon
from mxnet.gluon import nn
from layers import (Split2D, 
                    InvConv1x1LU, 
                    InvConv1x1, 
                    Squeeze2D, 
                    ActNorm, 
                    ZeroConv2D, 
                    AffineCoupling, 
                    Flow, 
                    Layer,
                    gaussian_ll, 
                    gaussian_sample,
                    split,
                    abs_log)


class Glow(nn.HybridBlock):
    ''' The Glow architecture '''

    def __init__(self, image_shape, K=3, L=2, affine=True, filter_size=512, temp=0.8, n_bits=8):
        super().__init__()

        in_channels = image_shape[0]
        self.flow_net = FlowNet(in_channels, K, L, affine, filter_size, temp)
        z_shapes = self.get_z_shapes(image_shape, L)
        out_channels, h, w = z_shapes[-1]

        self.learnable_prior = ZeroConv2D(out_channels, out_channels*2)
        self.prior = self.params.get('prior', shape=(1, out_channels, h, w), differentiable=False, init=mx.init.Zero())
        self.prior.initialize()
        self.i = 1

        self.n_bits = n_bits

    def get_z_shapes(self, image_shape, L):
        ''' Gets the spatial and channel dimensions throughout the network forward pass. Sanity check to ensure
        that the model dimensions are also the same! '''
        z_shapes = []
        c, h, w = image_shape

        # Each block halves the spatial dimensions and doubles the number of channels
        for _ in range(L - 1):
            h, w = h // 2, w // 2
            c = c * 2
            z_shapes.append((c, h, w))

        h, w = h // 2, w // 2
        z_shapes.append((c * 4, h, w))

        return z_shapes

    def sample_prior(self, n_samples, ctx):
        '''  Sample from learnable prior ! '''
        h = nd.repeat(self.prior.data(ctx), repeats=n_samples, axis=0)
        h = self.learnable_prior(h)
        mean, log_sd = split(h)

        return mean, log_sd

    def add_uniform_noise(self, x):
        ''' Add uniform noise to the input to avoid arbitrary large nll for continuous data. 
        See Theis (2016) for discussion. Natural images stored as ints of 2^8 bits = 256 pixels. '''

        bs, c, h, w = x.shape
        n_bins = 2 ** self.n_bits  # discretization level of the data
        n_pixels = c * h * w  # dimensionality of the input

        noise = ndarray.random.uniform_like(low=0, high=1.0/n_bins, data=x)
        x = x + noise

        # constant term when adding noise. Glow p.1
        objective = -math.log(n_bins) * n_pixels * nd.ones(bs, x.context)

        return x, objective

    def forward(self, x_in):
        ''' This is an encoding from a sample x onto the latent space z '''
        bs, c, h, w = x_in.shape
        n_pixels = c * h * w

        # This objective is the constant term of the LL when adding noise
        x_in, objective = self.add_uniform_noise(x_in)
        z_list, log_det = self.flow_net(x_in)
        # sample a mean and variance p. pixel
        mean, log_sd = self.sample_prior(bs, x_in.context)

        # With the change of variable in log space we add the determinant simply to the LL
        objective = objective + log_det

        # Compute the LL of the final transformation z_l given sampled mean and variance
        nll = objective + gaussian_ll(mean, log_sd, z_list[-1])
        nll = nll * -1

        # The average negative log likelihood divided by number of pixels => bits per dimension!
        bpd = (nll) / (math.log(2.0) * n_pixels)

        return z_list, nll, bpd

    def reverse(self, z=None, n_samples=2):
        ''' Decode from latent space or generate new sample '''
        ctx = self.learnable_prior.params.list_ctx()[0] # take one of the gpus / cpu available

        with mx.autograd.predict_mode():
            # sample z if not given (difference between reconstructing input and generating new sample)
            if z is None:
                mean, log_sd = self.sample_prior(n_samples, ctx)
                z = gaussian_sample(mean, log_sd)
            elif isinstance(z, list):
                z = z[-1]  # take only last forward pass

            x = self.flow_net.reverse(z)

        return x
        

class FlowNet(Layer):
    ''' A FlowNet does the following operations: Squeeze <-> Flow <-> Split. L is the number of Flow blocks. 
    K is the number of Squeeze, Flow, Split steps in one Block '''

    def __init__(self, in_channels, K, L, affine=True, filter_size=512, temp=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.K = K
        self.L = L
        self.affine = affine
        self.temp = temp
        self.filter_size = filter_size
        self._init_network()

    def _init_network(self):
        self.layers = nn.HybridSequential()
        ch = self.in_channels

        with self.name_scope():
            for i in range(self.L):
                ch *= 4
                self.layers.add(Squeeze2D())

                for _ in range(self.K):
                    self.layers.add(Flow(ch, self.affine, self.filter_size))

                # All Blocks split exept the last one
                if i < (self.L - 1):
                    ch = ch // 2
                    self.layers.add(Split2D(ch))

    def forward(self, x_in):
        ''' Forward pass through the entire network '''
        log_det = 0
        z_i = x_in
        z_list = []
        for layer in self.layers:
            z_i, det = layer(z_i)
            log_det = log_det + det
            z_list.append(z_i)

        return z_list, log_det

    def reverse(self, x_out):
        ''' Backward (reverse) pass through the entire network '''
        x_in = x_out
        for layer in reversed(self.layers):
            if isinstance(layer, Split2D):
                # sample with temperature from gaussian
                x_in = layer.reverse(x_in, self.temp)

            else:
                x_in = layer.reverse(x_in)
                
        return x_in