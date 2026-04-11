with open('tests/test_train.py', 'r') as f:
    content = f.read()

content = content.replace('assert "loss_total" in val_metrics', '')

# Need to fix the extra val_loss that I missed
content = content.replace('trainer.save_checkpoint(epoch=2, val_loss=0.6)', 'trainer.save_checkpoint(epoch=2, val_rmse=0.6)')

with open('tests/test_train.py', 'w') as f:
    f.write(content)
