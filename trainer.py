import paddle
import os
import time
import shutil
import pickle
from tqdm import tqdm
from tensorboard_logger import configure, log_value
from model import RecurrentAttention
from utils import AverageMeter


class Trainer:
    """A Recurrent Attention Model trainer.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args:
            config: object containing command line arguments.
            data_loader: A data iterator.
        """
        self.config = config
        if config.use_gpu and paddle.device.cuda.device_count() >= 1:
            self.device = str('cuda').replace('cuda', 'gpu')
        else:
            self.device = str('cpu').replace('cuda', 'gpu')
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size
        self.std = config.std
        self.M = config.M
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.sampler.indices)
            self.num_valid = len(self.valid_loader.sampler.indices)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = 10
        self.num_channels = 1
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.0
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = 'ram_{}_{}x{}_{}'.format(config.num_glimpses,
            config.patch_size, config.patch_size, config.glimpse_scale)
        self.plot_dir = './plots/' + self.model_name + '/'
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)
        self.model = RecurrentAttention(self.patch_size, self.num_patches,
            self.glimpse_scale, self.num_channels, self.loc_hidden, self.
            glimpse_hidden, self.std, self.hidden_size, self.num_classes)
        self.model.to(self.device)
        self.optimizer = paddle.optimizer.Adam(parameters=self.model.
            parameters(), learning_rate=self.config.init_lr, weight_decay=0.0)
        tmp_lr = paddle.optimizer.lr.ReduceOnPlateau(mode='min', patience=
            self.lr_patience, learning_rate=self.optimizer.get_lr())
        self.optimizer.set_lr_scheduler(tmp_lr)
        self.scheduler = tmp_lr

    def reset(self):
        out_0 = paddle.zeros(shape=[self.batch_size, self.hidden_size],
            dtype='float32')
        out_0.stop_gradient = not True
        h_t = out_0
        l_t = paddle.empty(shape=[self.batch_size, 2], dtype='float32'
            ).uniform_(min=-1, max=1).to(self.device)
        l_t.stop_gradient = not True
        return h_t, l_t

    def train(self):
        """Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        if self.resume:
            self.load_checkpoint(best=False)
        print('\n[*] Train on {} samples, validate on {} samples'.format(
            self.num_train, self.num_valid))
        for epoch in range(self.start_epoch, self.epochs):
            print('\nEpoch: {}/{} - LR: {:.6f}'.format(epoch + 1, self.
                epochs, self.optimizer.param_groups[0]['lr']))
            train_loss, train_acc = self.train_one_epoch(epoch)
            valid_loss, valid_acc = self.validate(epoch)
            self.scheduler.step(-valid_acc)
            is_best = valid_acc > self.best_valid_acc
            msg1 = 'train loss: {:.3f} - train acc: {:.3f} '
            msg2 = '- val loss: {:.3f} - val acc: {:.3f} - val err: {:.3f}'
            if is_best:
                self.counter = 0
                msg2 += ' [*]'
            msg = msg1 + msg2
            print(msg.format(train_loss, train_acc, valid_loss, valid_acc, 
                100 - valid_acc))
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print('[!] No improvement in a while, stopping training.')
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint({'epoch': epoch + 1, 'model_state': self.
                model.state_dict(), 'optim_state': self.optimizer.
                state_dict(), 'best_valid_acc': self.best_valid_acc}, is_best)

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
>>>>>>                self.optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                plot = False
                if epoch % self.plot_freq == 0 and i == 0:
                    plot = True
                self.batch_size = x.shape[0]
                h_t, l_t = self.reset()
                imgs = []
                imgs.append(x[0:9])
                locs = []
                log_pi = []
                baselines = []
                for t in range(self.num_glimpses - 1):
                    h_t, l_t, b_t, p = self.model(x, l_t, h_t)
                    locs.append(l_t[0:9])
                    baselines.append(b_t)
                    log_pi.append(p)
                h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last
                    =True)
                log_pi.append(p)
                baselines.append(b_t)
                locs.append(l_t[0:9])
                x = paddle.stack(x=baselines)
                perm_0 = list(range(x.ndim))
                perm_0[1] = 0
                perm_0[0] = 1
                baselines = x.transpose(perm=perm_0)
                x = paddle.stack(x=log_pi)
                perm_1 = list(range(x.ndim))
                perm_1[1] = 0
                perm_1[0] = 1
                log_pi = x.transpose(perm=perm_1)
                predicted = (paddle.max(x=log_probas, axis=1), paddle.
                    argmax(x=log_probas, axis=1))[1]
                R = (predicted.detach() == y).astype(dtype='float32')
                R = R.unsqueeze(axis=1).repeat(1, self.num_glimpses)
                loss_action = paddle.nn.functional.nll_loss(input=
                    log_probas, label=y)
                loss_baseline = paddle.nn.functional.mse_loss(input=
                    baselines, label=R)
                adjusted_reward = R - baselines.detach()
                loss_reinforce = paddle.sum(x=-log_pi * adjusted_reward, axis=1
                    )
                loss_reinforce = paddle.mean(x=loss_reinforce, axis=0)
                loss = loss_action + loss_baseline + loss_reinforce * 0.01
                correct = (predicted == y).astype(dtype='float32')
                acc = 100 * (correct.sum() / len(y))
                losses.update(loss.item(), x.shape[0])
                accs.update(acc.item(), x.shape[0])
                loss.backward()
                self.optimizer.step()
                toc = time.time()
                batch_time.update(toc - tic)
                pbar.set_description('{:.1f}s - loss: {:.3f} - acc: {:.3f}'
                    .format(toc - tic, loss.item(), acc.item()))
                pbar.update(self.batch_size)
                if plot:
                    imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
                    locs = [l.cpu().data.numpy() for l in locs]
                    pickle.dump(imgs, open(self.plot_dir + 'g_{}.p'.format(
                        epoch + 1), 'wb'))
                    pickle.dump(locs, open(self.plot_dir + 'l_{}.p'.format(
                        epoch + 1), 'wb'))
                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + i
                    log_value('train_loss', losses.avg, iteration)
                    log_value('train_acc', accs.avg, iteration)
            return losses.avg, accs.avg

    @paddle.no_grad()
    def validate(self, epoch):
        """Evaluate the RAM model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()
        for i, (x, y) in enumerate(self.valid_loader):
            x, y = x.to(self.device), y.to(self.device)
            x = x.repeat(self.M, 1, 1, 1)
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()
            log_pi = []
            baselines = []
            for t in range(self.num_glimpses - 1):
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)
                baselines.append(b_t)
                log_pi.append(p)
            h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
            log_pi.append(p)
            baselines.append(b_t)
            x = paddle.stack(x=baselines)
            perm_2 = list(range(x.ndim))
            perm_2[1] = 0
            perm_2[0] = 1
            baselines = x.transpose(perm=perm_2)
            x = paddle.stack(x=log_pi)
            perm_3 = list(range(x.ndim))
            perm_3[1] = 0
            perm_3[0] = 1
            log_pi = x.transpose(perm=perm_3)
            log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
            log_probas = paddle.mean(x=log_probas, axis=0)
            baselines = baselines.view(self.M, -1, baselines.shape[-1])
            baselines = paddle.mean(x=baselines, axis=0)
            log_pi = log_pi.view(self.M, -1, log_pi.shape[-1])
            log_pi = paddle.mean(x=log_pi, axis=0)
            predicted = (paddle.max(x=log_probas, axis=1), paddle.argmax(x=
                log_probas, axis=1))[1]
            R = (predicted.detach() == y).astype(dtype='float32')
            R = R.unsqueeze(axis=1).repeat(1, self.num_glimpses)
            loss_action = paddle.nn.functional.nll_loss(input=log_probas,
                label=y)
            loss_baseline = paddle.nn.functional.mse_loss(input=baselines,
                label=R)
            adjusted_reward = R - baselines.detach()
            loss_reinforce = paddle.sum(x=-log_pi * adjusted_reward, axis=1)
            loss_reinforce = paddle.mean(x=loss_reinforce, axis=0)
            loss = loss_action + loss_baseline + loss_reinforce * 0.01
            correct = (predicted == y).astype(dtype='float32')
            acc = 100 * (correct.sum() / len(y))
            losses.update(loss.item(), x.shape[0])
            accs.update(acc.item(), x.shape[0])
            if self.use_tensorboard:
                iteration = epoch * len(self.valid_loader) + i
                log_value('valid_loss', losses.avg, iteration)
                log_value('valid_acc', accs.avg, iteration)
        return losses.avg, accs.avg

    @paddle.no_grad()
    def test(self):
        """Test the RAM model.

        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0
        self.load_checkpoint(best=self.best)
        for i, (x, y) in enumerate(self.test_loader):
            x, y = x.to(self.device), y.to(self.device)
            x = x.repeat(self.M, 1, 1, 1)
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()
            for t in range(self.num_glimpses - 1):
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)
            h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
            log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
            log_probas = paddle.mean(x=log_probas, axis=0)
            pred = log_probas.data.max(1, keepdim=True)[1]
            correct += pred.equal(y=y.data.view_as(pred)).cpu().sum()
        perc = 100.0 * correct / self.num_test
        error = 100 - perc
        print('[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(correct,
            self.num_test, perc, error))

    def save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        paddle.save(obj=state, path=ckpt_path)
        if is_best:
            filename = self.model_name + '_model_best.pth.tar'
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    def load_checkpoint(self, best=False):
        """Load the best copy of a model.

        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        print('[*] Loading model from {}'.format(self.ckpt_dir))
        filename = self.model_name + '_ckpt.pth.tar'
        if best:
            filename = self.model_name + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = paddle.load(path=ckpt_path)
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.set_state_dict(state_dict=ckpt['model_state'])
        self.optimizer.set_state_dict(state_dict=ckpt['optim_state'])
        if best:
            print(
                '[*] Loaded {} checkpoint @ epoch {} with best valid acc of {:.3f}'
                .format(filename, ckpt['epoch'], ckpt['best_valid_acc']))
        else:
            print('[*] Loaded {} checkpoint @ epoch {}'.format(filename,
                ckpt['epoch']))
