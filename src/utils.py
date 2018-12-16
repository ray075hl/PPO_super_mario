def adjust_learning_rate(optimizer, initial_lr, max_update_times, current_update_times):
    lr = initial_lr*(1 - 1.0*current_update_times/max_update_times)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

