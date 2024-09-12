import os
import torch
from tqdm import tqdm
from models.loss import CE_Loss, Dice_loss, Focal_Loss
from utils.initialization import get_lr
from utils.metrics import f_score

def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss, total_f_score = 0, 0
    val_loss, val_f_score = 0, 0

    # Only display progress bar and print information on the main process
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch

        # Move data to GPU if available
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs, pngs, labels, weights = imgs.cuda(local_rank), pngs.cuda(local_rank), labels.cuda(local_rank), weights.cuda(local_rank)

        optimizer.zero_grad()

        if not fp16:
            # Standard precision training
            outputs = model_train(imgs)
            loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes) if focal_loss else CE_Loss(outputs, pngs, weights, num_classes=num_classes)
            if dice_loss:
                loss += Dice_loss(outputs, labels)
            with torch.no_grad():
                _f_score = f_score(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            # Mixed precision training
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(imgs)
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes) if focal_loss else CE_Loss(outputs, pngs, weights, num_classes=num_classes)
                if dice_loss:
                    loss += Dice_loss(outputs, labels)
                with torch.no_grad():
                    _f_score = f_score(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(total_loss=total_loss / (iteration + 1), f_score=total_f_score / (iteration + 1), lr=get_lr(optimizer))
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch

        # Move data to GPU if available
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs, pngs, labels, weights = imgs.cuda(local_rank), pngs.cuda(local_rank), labels.cuda(local_rank), weights.cuda(local_rank)
            outputs = model_train(imgs)
            loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes) if focal_loss else CE_Loss(outputs, pngs, weights, num_classes=num_classes)
            if dice_loss:
                loss += Dice_loss(outputs, labels)
            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(val_loss=val_loss / (iteration + 1), f_score=val_f_score / (iteration + 1), lr=get_lr(optimizer))
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        # Log the losses for this epoch
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print(f'Epoch: {epoch + 1}/{Epoch}')
        print(f'Total Loss: {total_loss / epoch_step:.3f} || Val Loss: {val_loss / epoch_step_val:.3f}')

        # Save model at specified intervals and at the end
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, f'ep{epoch + 1:03d}-loss{total_loss / epoch_step:.3f}-val_loss{val_loss / epoch_step_val:.3f}.pth'))

        # Save best model based on validation loss
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        # Always save the last epoch model
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
