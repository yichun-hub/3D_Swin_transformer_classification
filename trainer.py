import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch
from tqdm import tqdm


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_acc = AverageMeter()
    with tqdm(loader, unit="batch") as tepoch:
        for idx, batch_data in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)

            target = target.unsqueeze(1).float()  # 將 target 從 [batch_size] 轉換成 [batch_size, 1]
            
            optimizer.zero_grad()
            # for param in model.parameters():
            #     param.grad = None
            with autocast(enabled=args.amp):
                logits = model(data)                
                loss = loss_func(logits, target)
                
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if args.distributed:
                loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                run_loss.update(
                    np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
                )
            else:
                run_loss.update(loss.item(), n=args.batch_size)

            # Compute accuracy
            y_pred = torch.sigmoid(logits)
            preds = (y_pred > 0.5).float()  # 轉換為0或1
            accuracy = (preds == target).float().mean()            
            run_acc.update(accuracy.item(), n=args.batch_size)

            if args.rank == 0:
                tepoch.set_postfix(
                    train_loss = "{:.4f}".format(run_loss.avg),
                    train_acc = "{:.4f}".format(run_acc.avg),
                    time = "{:.2f}s".format(time.time() - start_time))

            start_time = time.time()
        # for param in model.parameters():
        #     param.grad = None
    return run_loss.avg, run_acc.avg

def val_epoch_category(model, loader, epoch, loss_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    running_vloss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            target = target.unsqueeze(1).float()  # 將 target 從 [batch_size] 轉換成 [batch_size, 1]

            with autocast(enabled=args.amp):  #AMP: Automatic mixed precision: 節省顯存，加快速度
                if model_inferer is not None:
                    logits = model_inferer(data) # seg
                else:
                    logits = model(data) # category
            
            y_pred = torch.sigmoid(logits)
            #print(y_pred)

            if not logits.is_cuda:
                target = target.cpu()

            # Compute loss
            vloss = loss_func(logits, target)
            running_vloss += vloss

    
            # Get metric
            # Threshold prediction with arg max
            #prediction = logits.argmax(dim=-1) ## for multi class classification

            # Compute accuracy
            # for cross entropy
            #accuracy = (prediction == target).sum() / float(prediction.shape[0])
            
            preds = (y_pred > 0.5).float()  # 轉換為0或1
            accuracy = (preds == target).float().mean()            
            running_acc += accuracy

    avg_vloss = running_vloss / (idx + 1) ## length of loader
    avg_acc = running_acc / (idx + 1)
    #return run_acc.avg
    return avg_acc, avg_vloss

def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
    early_stopping_patience=15,
):
    train_writer = None
    test_writer = None
    if args.logdir is not None and args.rank == 0:
        train_writer = SummaryWriter(log_dir=os.path.join(args.logdir, 'train'))
        test_writer = SummaryWriter(log_dir=os.path.join(args.logdir, 'test'))
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    running_vloss = 0.0
    early_stopping_counter = 0
    previous_val_loss = None
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "Taining loss: {:.4f}".format(train_loss),
                "Taining accuracy: {:.4f}".format(train_acc),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and train_writer is not None:
            train_writer.add_scalar("Loss", train_loss, epoch)
            train_writer.add_scalar("Accuracy", train_acc, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc, val_avg_loss = val_epoch_category(
                model,
                val_loader,
                epoch=epoch,
                loss_func = loss_func,
                model_inferer=model_inferer, #None
                args=args,
                post_label=post_label, #None
                post_pred=post_pred, #None
            )           

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "| Val acc:",  val_avg_acc.data.cpu().numpy(),
                    "| Time: {:.2f}s".format(time.time() - epoch_time),
                    "| Val loss:", val_avg_loss.data.cpu().numpy()
                )
                if test_writer is not None:
                    test_writer.add_scalar("Accuracy", val_avg_acc, epoch)
                    test_writer.add_scalar("Loss", val_avg_loss, epoch)               

                
                ## save the model based on best val accuracy
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    #early_stopping_counter = 0
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )

                ## early stopping based on val loss
                if previous_val_loss is None:
                    previous_val_loss = val_avg_loss

                else:
                    if val_avg_loss < previous_val_loss:
                        previous_val_loss = val_avg_loss
                        early_stopping_counter = 0                
                    
                    else:
                        early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                # if b_new_best:
                #     print("Copying to model.pt new best model!!!!")
                #     shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
