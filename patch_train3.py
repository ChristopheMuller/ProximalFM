import re

with open('proximalfm/Train.py', 'r') as f:
    content = f.read()

# Fix checkpoint and train loops
content = content.replace('''    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "") -> None:
        path_dir = os.path.join(
            self.train_cfg["save_dir"], self.train_cfg.get("exp_name", "cate_run")
        )
        os.makedirs(path_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "epoch_times": self.epoch_times,
        }
        name = f"model_epoch_{epoch}{('_' + tag) if tag else ''}.pth"
        path = os.path.join(path_dir, name)
        torch.save(checkpoint, path)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(path_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"  → New best model (val NLL: {val_loss:.4f})")''', '''    def save_checkpoint(self, epoch: int, val_rmse: float, tag: str = "") -> None:
        path_dir = os.path.join(
            self.train_cfg["save_dir"], self.train_cfg.get("exp_name", "cate_run")
        )
        os.makedirs(path_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_rmse, # keep key for compatibility
            "epoch_times": self.epoch_times,
        }
        name = f"model_epoch_{epoch}{('_' + tag) if tag else ''}.pth"
        path = os.path.join(path_dir, name)
        torch.save(checkpoint, path)

        if val_rmse < self.best_val_loss:
            self.best_val_loss = val_rmse
            best_path = os.path.join(path_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"  → New best model (val RMSE: {val_rmse:.4f})")''')


content = content.replace('''        early_stop_count = 0
        best_val_loss = getattr(self, "best_val_loss", float("inf"))

        reset_freq = self.data_cfg.get("reset_freq_train", -1)

        epoch = self.start_epoch
        last_val_nll = float("inf")
        while epoch < self.train_cfg["epochs"]:

            if hasattr(self.train_loader.dataset, "set_reset_state") and reset_freq > 0:
                self.train_loader.dataset.set_reset_state(epoch // reset_freq)

            train_metrics = self.train_epoch(epoch)
            val_metrics = None
            if epoch % self.train_cfg.get("val_freq", 1) == 0:
                val_metrics = self.validate(epoch)
                last_val_nll = val_metrics["loss_gnll"]

            # Extract metrics for console logging
            train_nll = train_metrics["loss_gnll"]
            train_mse = train_metrics["loss_mse"]
            train_total = train_metrics["loss_total"]
            val_gnll = val_metrics["loss_gnll"] if val_metrics else float("nan")

            logger.info(
                f"Epoch {epoch+1}/{self.train_cfg['epochs']}  "
                f"train_nll={train_nll:.4f}  train_rmse={train_mse:.4f}  "
                f"train_total={train_total:.4f}"
            )
            self.log_metrics(epoch, train_metrics, val_metrics)

            if (epoch + 1) % self.train_cfg.get("save_freq", 10) == 0 or epoch == 0:
                # Save using the last recorded validation NLL metric
                self.save_checkpoint(epoch + 1, last_val_nll)

            if val_metrics:
                if val_gnll < best_val_loss:
                    best_val_loss = val_gnll
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    if early_stop_count >= self.train_cfg["early_stop"]:
                        logger.info(f"Early stopping after {epoch+1} epochs.")
                        break
            epoch += 1

        self.total_training_time = time.time() - training_start
        # Final checkpoint uses the last recorded val_nll
        self.save_checkpoint(epoch + 1, last_val_nll, tag="final")''', '''        early_stop_count = 0
        best_val_loss = getattr(self, "best_val_loss", float("inf"))

        reset_freq = self.data_cfg.get("reset_freq_train", -1)

        epoch = self.start_epoch
        last_val_rmse = float("inf")
        while epoch < self.train_cfg["epochs"]:

            if hasattr(self.train_loader.dataset, "set_reset_state") and reset_freq > 0:
                self.train_loader.dataset.set_reset_state(epoch // reset_freq)

            train_metrics = self.train_epoch(epoch)
            val_metrics = None
            if epoch % self.train_cfg.get("val_freq", 1) == 0:
                val_metrics = self.validate(epoch)
                last_val_rmse = val_metrics["rmse"]

            # Extract metrics for console logging
            train_nll = train_metrics["loss_gnll"]
            train_mse = train_metrics["loss_mse"]
            train_total = train_metrics["loss_total"]
            val_rmse = val_metrics["rmse"] if val_metrics else float("nan")

            logger.info(
                f"Epoch {epoch+1}/{self.train_cfg['epochs']}  "
                f"train_nll={train_nll:.4f}  train_rmse={train_mse:.4f}  "
                f"train_total={train_total:.4f}"
            )
            self.log_metrics(epoch, train_metrics, val_metrics)

            if (epoch + 1) % self.train_cfg.get("save_freq", 10) == 0 or epoch == 0:
                # Save using the last recorded validation RMSE metric
                self.save_checkpoint(epoch + 1, last_val_rmse)

            if val_metrics:
                if val_rmse < best_val_loss:
                    best_val_loss = val_rmse
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    if early_stop_count >= self.train_cfg["early_stop"]:
                        logger.info(f"Early stopping after {epoch+1} epochs.")
                        break
            epoch += 1

        self.total_training_time = time.time() - training_start
        # Final checkpoint uses the last recorded val_rmse
        self.save_checkpoint(epoch + 1, last_val_rmse, tag="final")''')

with open('proximalfm/Train.py', 'w') as f:
    f.write(content)
