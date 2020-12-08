import mxnet as mx

class LossBuffer(object):
    def __init__(self):
        self._loss = None

    def new_loss(self, loss):
        ret = self._loss
        self._loss = loss
        return ret

    @property
    def loss(self):
        return self._loss

def get_context(use_gpu=-1):
    # Load Context and GPUs for distributed training
    gpus = mx.test_utils.list_gpus() if use_gpu > -1 else []
    n_gpus = len(gpus)
    if gpus:
        ctx = [mx.gpu(use_gpu)]
    else:
        ctx = [mx.cpu()]

    return ctx

def set_context(net, ctx):
    # Reinitialize the context for all parameters to allow training (and splitting) on (multiple) gpus
    if ctx[0].device_typeid == 2: # if ctx is gpu
        parameters = net.collect_params()
        for key in parameters.keys():
            param = parameters[key]
            param.reset_ctx(ctx)
    else:
        # set context for all initializations
        mx.test_utils.set_default_context(ctx[0])

    return net