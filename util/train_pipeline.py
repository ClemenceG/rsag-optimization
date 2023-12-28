import torch
import numpy as np
import pandas as pd
from .misc import calc_accuracy
from torch.optim import lr_scheduler

class HPScheduler:

    def step_scheduler(optimizer, tot_iters=30, start_f=1, end_f=0.5):
        """
        Linearly decrease the factor from start_f to end_f over tot_iters
        """
        return lr_scheduler.StepLR(optimizer, step_size=tot_iters, gamma=(end_f-start_f)/tot_iters)

    def lambda_scheduler(optimizer, base=0.1, factor=0.01):
        """
        factor: float
        base: float
        base/(1+factor*epoch)
        """
        l = lambda epoch: base/(1+factor*epoch)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=l)


def train_model(
                model,
                loss_function,
                optimizer,
                loaders,
                device='cpu',
                verbose=True,
                save_path=None,
                log_path=None,
                n_epochs=200,
                print_every=1
                ):
    log = {}
    log['loss'], log['accuracy'] = [], []
    log['v_loss'], log['v_accuracy'] = [], []
    
    log['v_loss_std'], log['v_accuracy_std'] = [], []
    log['loss_std'], log['accuracy_std'] = [], []

    best_acc = 0.0
    
    model.to(device)
    model.train()
    for epoch in range(0,n_epochs):
        print(f'Starting Epoch {epoch+1}')

        current_loss, total_acc = [], []
        v_loss, v_acc = [], []

        for data, targets in loaders['train']:
            # inputs, targets = data
            # inputs, targets = inputs.float(), targets.float()
            # targets = targets.reshape((targets.shape[0], 1))
            
            # Copy data and targets to GPU
            data = data.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()

            outputs = model(data)

            # Calculate the loss
            loss = loss_function(outputs, targets)
            # current_loss += loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            current_loss.append(loss.item())
            total_acc.append(calc_accuracy(outputs, targets))
            
        # Validation
        model.eval()

        for data, targets in loaders['valid']:
            data = data.to(device)
            targets = targets.to(device)
            
            outputs = model(data)

            loss = loss_function(outputs, targets)
            v_loss.append(loss.item())
            v_acc.append(calc_accuracy(outputs, targets))        

            # if i%10 == 0:
            #     print(f'Loss after mini-batch %5d: %.3f'%(i+1, current_loss/500))
            #     current_loss = 0.0
        print(f'Epoch {epoch+1} finished')
        # current_loss /= len(loaders['train'])
        # total_acc /= len(loaders['train'])
        # print('loss {:.4f}'.format(current_loss))
        # print('Accuracy:  {:.4f}'.format(total_acc))

        update_log(log, current_loss, total_acc, v_loss, v_acc, len(loaders['train']), len(loaders['valid']))

        if verbose:
            if epoch%print_every == 0:
                print('Epoch {}/{}'.format(epoch+1, n_epochs))
                print('-' * 10)
                print('Loss {:.4f}'.format(log['loss'][-1]))
                print('Accuracy:  {:.4f}'.format(log['accuracy'][-1]))
                print('Validation Loss {:.4f}'.format(log['v_loss'][-1]))
                print('Validation Accuracy:  {:.4f}'.format(log['v_accuracy'][-1]))
        
        if len(log['v_accuracy']) > 1 and (np.abs(log['v_accuracy'][-1]-log['v_accuracy'][-2])<0.1):
            print('Early stopping at epoch %d'%epoch)
            break

        if log['v_accuracy'][-1] > best_acc:
            best_acc = log['v_accuracy'][-1]
        
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print('Model saved to %s'%save_path)

    if log_path is not None:
        df = pd.DataFrame.from_dict(log)
        df.to_csv(log_path)
        print('Log saved to %s'%log_path)

    print("Training has completed")
    return log, best_acc

def update_log(log, current_loss, total_acc, v_loss, v_acc, train_len, valid_len):
        log['loss_std'].append(np.std(current_loss))
        log['accuracy_std'].append(np.std(total_acc))
        current_loss = sum(current_loss)/train_len
        total_acc = sum(total_acc)/train_len
        log['loss'].append(current_loss)
        log['accuracy'].append(total_acc)

        
        log['v_loss_std'].append(np.std(v_loss))
        log['v_accuracy_std'].append(np.std(v_acc))
        v_loss = sum(v_loss)/valid_len
        v_acc = sum(v_acc)/valid_len
        log['v_loss'].append(v_loss)
        log['v_accuracy'].append(v_acc)

        return