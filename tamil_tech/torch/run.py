import time
from tamil_tech.torch.utils import *
import os
import time

import torch

from tamil_tech.torch.loss import cal_performance
from tamil_tech.torch.utils import IGNORE_ID

class Solver(object):
    """
    """

    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer

        # Low frame rate feature
        self.LFR_m = args.LFR_m
        self.LFR_n = args.LFR_n

        # Training config
        self.epochs = args.epochs
        self.label_smoothing = args.label_smoothing
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.visdom = args.visdom
        self.visdom_lr = args.visdom_lr
        self.visdom_epoch = args.visdom_epoch
        self.visdom_id = args.visdom_id
        if self.visdom:
            from visdom import Visdom
            self.vis = Visdom(env=self.visdom_id)
            self.vis_opts = dict(title=self.visdom_id,
                                 ylabel='Loss', xlabel='Epoch',
                                 legend=['train loss', 'cv loss'])
            self.vis_window = None
            self.vis_epochs = torch.arange(1, self.epochs + 1)
            self.optimizer.set_visdom(self.visdom_lr, self.vis)

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1,
                                                self.LFR_m, self.LFR_n,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1,
                                                self.LFR_m, self.LFR_n,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss),
                           file_path)
                print("Find better validated model, saving to %s" % file_path)

            # visualizing loss using visdom
            if self.visdom:
                x_axis = self.vis_epochs[0:epoch + 1]
                y_axis = torch.stack(
                    (self.tr_loss[0:epoch + 1], self.cv_loss[0:epoch + 1]), dim=1)
                if self.vis_window is None:
                    self.vis_window = self.vis.line(
                        X=x_axis,
                        Y=y_axis,
                        opts=self.vis_opts,
                    )
                else:
                    self.vis.line(
                        X=x_axis.unsqueeze(0).expand(y_axis.size(
                            1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                        Y=y_axis,
                        win=self.vis_window,
                        update='replace',
                    )

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # visualizing loss using visdom
        if self.visdom_epoch and not cross_valid:
            vis_opts_epoch = dict(title=self.visdom_id + " epoch " + str(epoch),
                                  ylabel='Loss', xlabel='Epoch')
            vis_window_epoch = None
            vis_iters = torch.arange(1, len(data_loader) + 1)
            vis_iters_loss = torch.Tensor(len(data_loader))

        for i, (data) in enumerate(data_loader):
            padded_input, padded_target, input_lengths, label_lengths = data
            padded_input = padded_input.cuda()
            input_lengths = input_lengths.cuda()
            padded_target = padded_target.cuda()
            pred, gold = self.model(padded_input, input_lengths, padded_target)
            loss, n_correct = cal_performance(pred, gold,
                                              smoothing=self.label_smoothing)
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            non_pad_mask = gold.ne(IGNORE_ID)
            n_word = non_pad_mask.sum().item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)

            # visualizing loss using visdom
            if self.visdom_epoch and not cross_valid:
                vis_iters_loss[i] = loss.item()
                if i % self.print_freq == 0:
                    x_axis = vis_iters[:i+1]
                    y_axis = vis_iters_loss[:i+1]
                    if vis_window_epoch is None:
                        vis_window_epoch = self.vis.line(X=x_axis, Y=y_axis,
                                                         opts=vis_opts_epoch)
                    else:
                        self.vis.line(X=x_axis, Y=y_axis, win=vis_window_epoch,
                                      update='replace')

        return total_loss / (i + 1)

def train_one_epoch(model, device, train_loader, criterion, optimizer, scheduler, epoch, experiment):
    model.train()
    loss, avg_cer, avg_wer = 0, 0, 0
    data_len = len(train_loader.dataset)
    test_cer, test_wer = [], []

    start_time = time.time()

    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data 
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step()
        scheduler.step()

        if batch_idx % 100 == 0 or batch_idx == data_len:
            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1).type(torch.IntTensor), labels, label_lengths)

            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
                    
            avg_cer = sum(test_cer)/len(test_cer)
            avg_wer = sum(test_wer)/len(test_wer)

            print('Epoch: {} [{:4}/{} ({:.0f}%)] Loss: {:.4f} CER: {:.4f} WER: {:.4f} - Duration: {:.2f} minutes'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                100. * batch_idx / len(train_loader), loss.item(), avg_cer, avg_wer, ((time.time()-start_time)/60)))
            
            if experiment is not None:
              experiment.add_scalar('training_loss_in_steps', loss.item(), epoch * len(train_loader) + batch_idx)
              experiment.add_scalar('learning_rate_in_steps', np.array(scheduler.get_last_lr()), epoch * len(train_loader) + batch_idx)

    if experiment is not None:
      experiment.add_scalar('training_CER', avg_cer, epoch)
      experiment.add_scalar('training_WER', avg_wer, epoch)
      
      experiment.add_scalar('training_loss_in_epochs', loss.item(), epoch)
      experiment.add_scalar('learning_rate_in_epochs', np.array(scheduler.get_last_lr()), epoch)
    
    print(f"Epoch Duration: {(time.time()-start_time)/60:.2f} minutes")

    return loss

def test_one_epoch(model, device, test_loader, criterion, epoch, experiment, mode):
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
      for i, _data in enumerate(test_loader):
        spectrograms, labels, input_lengths, label_lengths = _data 
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        test_loss += loss.item() / len(test_loader)

        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1).type(torch.IntTensor), labels, label_lengths)

        for j in range(len(decoded_preds)):
            test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
                
    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)

    print('Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))

    if experiment is not None:
      experiment.add_scalar(mode+'_loss', test_loss, epoch)
      experiment.add_scalar(mode+'_CER', avg_cer, epoch)
      experiment.add_scalar(mode+'_WER', avg_wer, epoch)

    return test_loss

def train(epochs, model, device, train_loader, dev_loader, criterion, optimizer, scheduler, experiment, validate=True, checkpoint=False, checkpoint_path='/content/drive/My Drive/models', model_name='tamil_asr_new'):
  best_prec = 5

  print("Training...")

  for epoch in range(1, epochs + 1):
    # empty GPU cache
    torch.cuda.empty_cache()

    training_loss = train_one_epoch(model, device, train_loader, criterion, optimizer, scheduler, epoch, experiment)

    # empty gpu cache
    torch.cuda.empty_cache()

    print('\nEvaluating...')

    val_loss = test_one_epoch(model, device, dev_loader, criterion, epoch, experiment, mode='validation')

    torch.cuda.empty_cache()
    
    if checkpoint:
      # save as checkpoints
      is_best = (val_loss < best_prec)
    
      # if best validation loss, update best_prec and checkpoint
      if is_best:
        best_prec = min(val_loss, best_prec)
        
        save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
        }, filename=os.path.join(checkpoint_path, model_name + f'_{epoch}.pt'))