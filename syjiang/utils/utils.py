import torch 
import csv
import os

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def save_results(epoch_num, train_loss, val_loss, train_acc , val_acc, file_dir, file_name):
    os.makedirs(file_dir, exist_ok=True)
    path = os.path.join(file_dir, '{}.csv'.format(file_name))

    columns = ['epoch_num', 'train_loss', 'val_loss', 'train_acc' , 'val_acc']
    current_result = [epoch_num, train_loss, val_loss, train_acc , val_acc]

    if epoch_num == 1:
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerow(current_result)
    else:
        with open(path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(current_result)
