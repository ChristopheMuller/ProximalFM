with open('tests/test_train.py', 'r') as f:
    content = f.read()

content = content.replace(
'''    mock_out = torch.randn(B, 2, 2)
    trainer.model = MagicMock(return_value=mock_out)''',
'''    mock_out_mean = torch.randn(B, 2, 1)
    mock_out_var = torch.randn(B, 2, 1)
    trainer.model = MagicMock()
    trainer.model.estimate_CATE.return_value = (mock_out_mean, mock_out_var)''')

content = content.replace('trainer.save_checkpoint(epoch=1, val_loss=0.5)', 'trainer.save_checkpoint(epoch=1, val_rmse=0.5)')
content = content.replace('trainer.save_checkpoint(epoch=2, val_loss=0.3)', 'trainer.save_checkpoint(epoch=2, val_rmse=0.3)')
content = content.replace('assert torch.load(tmp_path / "cate_run/model_epoch_1.pth")["val_loss"] == 0.5', 'assert torch.load(tmp_path / "cate_run/model_epoch_1.pth")["val_loss"] == 0.5')
content = content.replace('assert torch.load(tmp_path / "cate_run/best_model.pth")["val_loss"] == 0.3', 'assert torch.load(tmp_path / "cate_run/best_model.pth")["val_loss"] == 0.3')

with open('tests/test_train.py', 'w') as f:
    f.write(content)
