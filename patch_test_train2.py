import re

with open('tests/test_train.py', 'r') as f:
    content = f.read()

content = content.replace(
'''    assert isinstance(val_metrics, dict)
    assert "loss_gnll" in val_metrics
    assert "rmse" in val_metrics''',
'''    assert isinstance(val_metrics, dict)
    assert "rmse" in val_metrics
    assert "loss_gnll" not in val_metrics''')

content = content.replace(
'''    # Test tracking of best val_loss
    assert trainer.best_val_loss == 0.5
    trainer.save_checkpoint(epoch=2, val_loss=0.6)
    assert trainer.best_val_loss == 0.5
    trainer.save_checkpoint(epoch=2, val_loss=0.3)
    assert trainer.best_val_loss == 0.3''',
'''    # Test tracking of best val_loss
    assert trainer.best_val_loss == 0.5
    trainer.save_checkpoint(epoch=2, val_rmse=0.6)
    assert trainer.best_val_loss == 0.5
    trainer.save_checkpoint(epoch=2, val_rmse=0.3)
    assert trainer.best_val_loss == 0.3''')

with open('tests/test_train.py', 'w') as f:
    f.write(content)
