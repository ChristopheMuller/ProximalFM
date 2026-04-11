with open('tests/test_model.py', 'r') as f:
    content = f.read()

content = content.replace("mean, var = model.estimate_CATE", "mean, log_var = model.estimate_CATE")
content = content.replace("assert var.shape == (batch_size, n_test, 1)", "assert log_var.shape == (batch_size, n_test, 1)")
content = content.replace("assert torch.all(var > 0)", "assert log_var.dim() == 3") # removing var > 0 check as it's log_var now

with open('tests/test_model.py', 'w') as f:
    f.write(content)
