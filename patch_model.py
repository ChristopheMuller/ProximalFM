with open('proximalfm/Model.py', 'r') as f:
    content = f.read()

content = content.replace(
    '''        mean = pred_test[..., 0:1]
        if return_variance:
            log_var = pred_test[..., 1:2]
            variance = torch.exp(log_var)
            return mean, variance''',
    '''        mean = pred_test[..., 0:1]
        if return_variance:
            log_var = pred_test[..., 1:2]
            return mean, log_var'''
)

with open('proximalfm/Model.py', 'w') as f:
    f.write(content)
