import torch
import numpy as np
import pandas as pd
from .misc import calc_accuracy

class TrainPipeline:

    def __init__(self,
                 model,
                 loss_function,
                 optimizer,
                 device='cpu',
                 verbose=True,
                 save_path=None,
                 log_path=None):
        self.model=model
        self.loss_function=loss_function
        self.optimizer=optimizer
        self.device=device
        self.verbose=verbose
        self.save_path=save_path
        self.log_path=log_path


    def train_model(self,
                    loaders,
                    n_epochs=5
                    ):
        log = {}
        log['loss'], log['accuracy'] = [], []
        log['v_loss'], log['v_accuracy'] = [], []
        
        log['v_loss_std'] = []
        log['v_accuracy_std'] = []

        
        log['loss_std'] = []
        log['accuracy_std'] = []

        for epoch in range(0,n_epochs):
            print(f'Starting Epoch {epoch+1}')

            current_loss, total_acc = [], []
            v_loss, v_acc = [], []

            for data, targets in loaders['train']:
                # inputs, targets = data
                # inputs, targets = inputs.float(), targets.float()
                # targets = targets.reshape((targets.shape[0], 1))
                
                # Copy data and targets to GPU
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()

                outputs = self.model(data)

                # Calculate the loss
                loss = self.loss_function(outputs, targets)
                # current_loss += loss

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                current_loss.append(loss.item())
                total_acc.append(calc_accuracy(outputs, targets))
                
            # Validation
            self.model.eval()

            for data, targets in loaders['valid']:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(data)

                loss = self.loss_function(outputs, targets)
                v_loss.append(loss.item())
                v_acc.append(self.calc_accuracy(outputs, targets))        

                # if i%10 == 0:
                #     print(f'Loss after mini-batch %5d: %.3f'%(i+1, current_loss/500))
                #     current_loss = 0.0
            print(f'Epoch {epoch+1} finished')
            # current_loss /= len(loaders['train'])
            # total_acc /= len(loaders['train'])
            # print('loss {:.4f}'.format(current_loss))
            # print('Accuracy:  {:.4f}'.format(total_acc))
            
            log['loss_std'].append(np.std(current_loss))
            log['accuracy_std'].append(np.std(total_acc))

            current_loss = sum(current_loss)/len(loaders['train'])
            total_acc = sum(total_acc)/len(loaders['train'])
            log['loss'].append(current_loss)
            log['accuracy'].append(total_acc)

            
            log['v_loss_std'].append(np.std(v_loss))
            log['v_accuracy_std'].append(np.std(v_acc))
            v_loss = sum(v_loss)/len(loaders['valid'])
            v_acc = sum(v_acc)/len(loaders['valid'])
            log['v_loss'].append(v_loss)
            log['v_accuracy'].append(v_acc)
            
            if self.verbose:
                print('Epoch {}/{}'.format(epoch+1, n_epochs))
                print('-' * 10)
                print('Loss {:.4f}'.format(current_loss))
                print('Accuracy:  {:.4f}'.format(total_acc))
                print('Validation Loss {:.4f}'.format(v_loss))
                print('Validation Accuracy:  {:.4f}'.format(v_acc))
            
        if self.save_path is not None:
            torch.save(self.model.state_dict(), self.save_path)
            print('Model saved to %s'%self.save_path)

        if self.log_path is not None:
            df = pd.DataFrame.from_dict(log)
            df.to_csv(self.log_path)
            print('Log saved to %s'%self.log_path)

        print("Training has completed")
        return log