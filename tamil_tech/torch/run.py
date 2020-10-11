import time
from tamil_tech.torch.utils import *

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

def train(epochs, model, device, train_loader, dev_loader, criterion, optimizer, scheduler, epoch, experiment, validate=True, checkpoint=False, checkpoint_path='/content/drive/My Drive/models', model_name='tamil_asr_new'):
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