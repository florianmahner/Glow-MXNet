import mxnet as mx
import numpy as np
from config import Config
from glow import Glow
from mxboard import SummaryWriter
from datetime import date
from utils import set_context, get_context, LossBuffer


def data_xform(data, n_bits=8):
    ''' Move channel axis to the beginning, cast to float32, and normalize to [0, 1]. '''
    data = mx.nd.moveaxis(data, 2, 0).astype('float32')
    data = mx.nd.tile(data, (3, 1, 1))
    data = data.reshape(1, *data.shape)
    data = mx.nd.pad(data, mode='constant', constant_value=0, pad_width=(0, 0, 0, 0, 2, 2, 2, 2))
    data = data / 255 - 0.5

    return data.squeeze()

def train(cfg):
    date_today = date.today().strftime("%b-%d-%Y")
    summary_writer = SummaryWriter(cfg.log_dir, flush_secs=5, filename_suffix=date_today)
    train_data = mx.gluon.data.vision.MNIST(train=True).transform_first(data_xform)
    train_loader = mx.gluon.data.DataLoader(train_data, shuffle=True, batch_size=cfg.batch_size)
    image_shape = train_data[0][0].shape

    # No initialization. Custom blocks encapsulate initialization and setting of data.
    net = Glow(image_shape, cfg.K, cfg.L, cfg.affine, cfg.filter_size, cfg.temp, cfg.n_bits)
    ctx = get_context(cfg.use_gpu)
    net = set_context(net, ctx)

    trainer = mx.gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': cfg.lr})
    n_samples = len(train_loader)
    update_interval = n_samples // 2 # store the loss with summary writer twice
    loss_buffer = LossBuffer()
    global_step = 1

    for epoch in range(1, cfg.n_epochs + 1):
        for idx, (batch, label) in enumerate(train_loader):
            print(f'Epoch {epoch} - Batch {idx}/{n_samples}', end='\r')

            data = mx.gluon.utils.split_and_load(batch, ctx)
            with mx.autograd.record():
                for X in data:
                    z_list, nll, bpd = net(X)
                    prev_loss = loss_buffer.new_loss(bpd.mean())

            loss_buffer.loss.backward()
            trainer.step(1)

            if prev_loss is not None and global_step % update_interval == 0:
                loss = prev_loss.asscalar()
                summary_writer.add_scalar(tag='bpd', value=loss, global_step=global_step)
    
            global_step += 1

        # Sample from latent space to generate random digit and reverse from latent
        if (epoch % cfg.plot_interval) == 0:
            x_generate = net.reverse()[0]
            x_generate = x_generate.reshape(1, *x_generate.shape)
            x_recon = net.reverse(z_list[-1])[0]
            x_recon = x_recon.reshape(1, *x_recon.shape)
            x_real = data[0][0].reshape(1, *data[0][0].shape)
            minim = -0.5
            maxim = 0.5
            x_generate = x_generate.clip(minim, maxim)
            x_generate += -minim 
            x_recon = x_recon.clip(minim, maxim)
            x_recon += -minim
            x_real += -minim
            
            img = mx.nd.concatenate([x_real, x_generate, x_recon], axis=0).asnumpy()
            summary_writer.add_image(tag='generations', image=img, global_step=global_step)

    summary_writer.close()



if __name__ == '__main__':
    cfg = Config()
    train(cfg)