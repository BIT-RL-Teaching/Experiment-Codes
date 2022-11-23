def normalize(a_list):
    mean_list, std_list = [], []
    for i in range(len(a_list)):
        array = a_list[i]
        mean_list.append(array.mean())
        std_list.append(array.std())
        a_list[i] = (array - array.mean()) / ((array.std() + 1e-4))  # sometimes helps
    return a_list, mean_list, std_list


def update_linear_schedule(optimizer1, optimizer2, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    # print(f'在第 {epoch} / {total_num_epochs} 次学习率衰减后， lr = {lr}')
    if optimizer1 is not None:
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
    if optimizer2 is not None:
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr