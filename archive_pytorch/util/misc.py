import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import json

def calc_accuracy(y_pred, labels):
    predicted_digits = y_pred.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == labels).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()

def plot_losses(losses: pd.DataFrame, 
                title='Losses', 
                std_devs=None, 
                save_path=None):

    if std_devs is not None:
        plt.errorbar(range(len(losses)), losses, yerr=std_devs)
    if len(losses.shape) >1:
        for i in range(losses.shape[1]):
            plt.plot(losses[:,i])
    else:
        plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    if save_path is not None:
        plt.savefig(save_path)

def plot_accuracies(accuracies: list[float],
                    title='Accuracies',
                    std_devs=None,
                    save_path=None):
    plt.plot(accuracies)
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.show()

def valid_train_stacked(
      title, valid_dict_losses, train_dict_losses, valid_dict_accs, train_dict_accs):

    valid_losses = pd.DataFrame.from_dict(valid_dict_losses)
    print(valid_losses)
    train_losses = pd.DataFrame.from_dict(train_dict_losses, orient='columns')

    valid_accs = pd.DataFrame.from_dict(valid_dict_accs, orient='columns')
    train_accs = pd.DataFrame.from_dict(train_dict_accs, orient='columns')

    fig, axs =  plt.subplots(2, 1)
    x_vals = range(len(list(valid_dict_losses.values())[0]))
    for name in valid_losses.columns:
    # ax.plot(df[df.name==name].year,df[df.name==name].weight,label=name)
      axs[0].plot(x_vals, valid_losses[name], label=f'{name} Training')
      axs[0].plot(x_vals, train_losses[name], label=f'{name} Validation')

       
      axs[1].plot(x_vals, valid_accs[name], label=f'{name} Training')
      axs[1].plot(x_vals, train_accs[name], label=f'{name} Validation')
      # axs[0].plot(x_vals, valid_losses[name], label='Validation')
      # axs[0].plot(x_vals, train_losses, label='Training')
    axs[0].set_title('Losses')
    axs[0].legend(bbox_to_anchor=(1,1), loc="upper left")

    # axs[1].legend(bbox_to_anchor=(2,1), loc="lower left")

    axs[1].set_title('Accuracies')
    fig.suptitle(title)

    # fig0.legend()
    # fig1.legend()
    
    plt.show()

  
# Read file
def read_json(file_path):
    with open(file_path, 'r') as f:
        json_f = json.load(f)
    return json_f

# res_folder = './exps/'
# dir_to_read = 'mlp_default/'
# path = os.path.join(res_folder, dir_to_read)


# train_losses, valid_losses = {}, {}
# train_accs, valid_accs = {},{}
# test_losses, test_accs = {}, {}



# for f in os.listdir(path):
#   fp = os.path.join(path, f)
#   json_f = read_json(fp)
#   # model_name = json_f['name']

#   model_name = f[:-5]
#   if model_name=='results': continue
#   print(model_name)
#   train_losses[model_name] = json_f['train_losses']
#   valid_losses[model_name] = json_f['valid_losses']
#   train_accs[model_name] = json_f['train_accs']
#   valid_accs[model_name] = json_f['valid_accs']
#   test_losses[model_name] = json_f['test_loss']
#   test_accs[model_name] = json_f['test_acc']

# print(train_losses)

# valid_train_stacked('MLP Activition Function Analysis', valid_losses, train_losses, valid_accs, train_accs)
