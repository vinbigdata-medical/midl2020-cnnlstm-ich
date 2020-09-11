import argparse
import os
import sys
import time

import apex
from apex import amp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss

from src.config import get_cfg
from src.data import RSNAHemorrhageDS3d, Qure500DS
from src.solver import make_lr_scheduler, make_optimizer
from src.modeling import ResNet3D, WeightedBCEWithLogitsLoss
from src.utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
            help="config yaml path")
    parser.add_argument("--load", type=str, default="",
            help="path to model weight")
    parser.add_argument("-ft", "--finetune", action="store_true",
        help="path to model weight")
    parser.add_argument("-m", "--mode", type=str, default="train",
        help="model running mode (train/valid/test)")
    parser.add_argument("--valid", action="store_true",
        help="enable evaluation mode for validation")
    parser.add_argument("--test", action="store_true",
        help="enable evaluation mode for testset")
    parser.add_argument("--test-qure", action="store_true",
        help="run test on QURE500 dataset")
    parser.add_argument("--tta", action="store_true",
        help="enable tta infer")
    parser.add_argument("-d", "--debug", action="store_true",
        help="enable debug mode for test")

    args = parser.parse_args()
    if args.valid:
        args.mode = "valid"
    elif args.test:
        args.mode = "test"

    return args


def build_model(cfg):
    model = ResNet3D
    return model(cfg)


def create_submission(pred_df, sub_fpath):
    imgid = pred_df["image"].values
    output = pred_df.loc[:, pred_df.columns[1:]].values
    data = [[iid]+[sub_o for sub_o in o] for iid, o in zip(imgid, output)]
    table_data = []
    for subdata in data:
        table_data.append([subdata[0]+'_any', subdata[1]])
        table_data.append([subdata[0]+'_intraparenchymal', subdata[2]])
        table_data.append([subdata[0]+'_intraventricular', subdata[3]])
        table_data.append([subdata[0]+'_subarachnoid', subdata[4]])
        table_data.append([subdata[0]+'_subdural', subdata[5]])
        table_data.append([subdata[0]+'_epidural', subdata[6]])
    df = pd.DataFrame(data=table_data, columns=['ID','Label'])
    df.to_csv(f'{sub_fpath}.csv', index=False)


def test_qure_dataset(_print, cfg, model, test_loader):
    # merge readers' labels
    reads = pd.read_csv("/home/nhannt/qure/reads.csv")
    reads1 = reads.loc[:, ["R1:ICH", "R1:IPH", "R1:IVH", "R1:SAH", "R1:SDH", "R1:EDH"]].values
    reads2 = reads.loc[:, ["R2:ICH", "R2:IPH", "R2:IVH", "R2:SAH", "R2:SDH", "R2:EDH"]].values
    reads3 = reads.loc[:, ["R3:ICH", "R3:IPH", "R3:IVH", "R3:SAH", "R3:SDH", "R3:EDH"]].values
    gtruth = []
    for r1, r2, r3 in zip(reads1, reads2, reads3):
        majority_voting = ((r1 + r2 + r3) / 3.) > 0.5
        gtruth.append(majority_voting.reshape(1, -1))
    gtruth = np.concatenate(gtruth, 0)
    gtruth_df = pd.concat([reads["name"], pd.DataFrame(gtruth)], axis=1)

    # switch to evaluate mode
    model.eval()

    ids = []
    probs = []
    tbar = tqdm(test_loader)

    with torch.no_grad():
        for i, (image, id_code) in enumerate(tbar):
            image = image.cuda()
            bsize, seq_len, c, h, w = image.size()
            image = image.view(bsize * seq_len, c, h, w)
            output = model(image, seq_len)
            output = torch.sigmoid(output)
            # Extract study-level prediction
            # output = output.mean(dim=0, keepdim=True)
            output, _ = torch.max(output, dim=0, keepdim=True)
            probs.append(output.cpu().numpy())
            ids += id_code

    probs = np.concatenate(probs, 0)
    pred_df = pd.concat([pd.Series(ids), pd.DataFrame(probs)], axis=1)
    pred_df.columns = ["name", "any",
                      "intraparenchymal", "intraventricular",
                      "subarachnoid", "subdural", "epidural"]

    pred_df = pred_df.set_index("name")
    gtruth_df = gtruth_df.set_index("name").reindex(pred_df.index)
    targets = gtruth_df.values
    preds = pred_df.values

    # record loss
    loss_array = [
        log_loss(targets[:, i], preds[:, i]) for i in range(6)]
    val_loss = np.average(loss_array, weights=cfg.LOSS.WEIGHTS)
    _print("Val. loss: %.5f - any: %.3f - intraparenchymal: %.3f - intraventricular: %.3f - subarachnoid: %.3f - subdural: %.3f - epidural: %.3f" % (
        val_loss, loss_array[0],
        loss_array[1], loss_array[2],
        loss_array[3], loss_array[4], loss_array[5]))
    # record AUC
    auc = roc_auc_score(targets, preds, average=None)
    _print("Val. AUC - ICH: %.4f - IPH: %.4f - IVH: %.4f - SAH: %.4f - SDH: %.4f - EDH: %.4f" % (
        auc[0], auc[1], auc[2], auc[3], auc[4], auc[5]))


def test_model(_print, cfg, model, test_loader):
    # switch to evaluate mode
    model.eval()

    ids = []
    probs = []
    tbar = tqdm(test_loader)

    with torch.no_grad():
        for i, (image, id_code) in enumerate(tbar):
            image = image.cuda()
            id_code = list(*zip(*id_code))
            bsize, seq_len, c, h, w = image.size()
            image = image.view(bsize * seq_len, c, h, w)
            output = model(image, seq_len)
            output = torch.sigmoid(output)
            probs.append(output.cpu().numpy())
            ids += id_code

    probs = np.concatenate(probs, 0)
    submit = pd.concat([pd.Series(ids), pd.DataFrame(probs)], axis=1)
    submit.columns = ["image", "any",
                      "intraparenchymal", "intraventricular",
                      "subarachnoid", "subdural", "epidural"]
    return submit


def valid_model(_print, cfg, model, valid_loader, valid_criterion):
    # switch to evaluate mode
    model.eval()

    preds = []
    targets = []
    tbar = tqdm(valid_loader)
    with torch.no_grad():
        for i, (image, target) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()
            bsize, seq_len, c, h, w = image.size()
            image = image.view(bsize * seq_len, c, h, w)
            target = target.view(-1, target.size(-1))
            output = model(image, seq_len)
            preds.append(output.cpu())
            targets.append(target.cpu())

    preds, targets = torch.cat(preds, 0), torch.cat(targets, 0)
    # record loss
    loss_tensor = valid_criterion(preds, targets)
    val_loss = loss_tensor.sum() / valid_criterion.class_weights.sum()
    any_loss = loss_tensor[0]
    intraparenchymal_loss = loss_tensor[1]
    intraventricular_loss = loss_tensor[2]
    subarachnoid_loss = loss_tensor[3]
    subdural_loss = loss_tensor[4]
    epidural_loss = loss_tensor[5]
    _print("Val. loss: %.5f - any: %.3f - intraparenchymal: %.3f - intraventricular: %.3f - subarachnoid: %.3f - subdural: %.3f - epidural: %.3f" % (
        val_loss, any_loss,
        intraparenchymal_loss, intraventricular_loss,
        subarachnoid_loss, subdural_loss, epidural_loss))
    # record AUC
    auc = roc_auc_score(targets[:, 1:].numpy(), preds[:, 1:].numpy(), average=None)
    _print("Val. AUC - intraparenchymal: %.3f - intraventricular: %.3f - subarachnoid: %.3f - subdural: %.3f - epidural: %.3f" % (
            auc[0], auc[1], auc[2], auc[3], auc[4]))
    return val_loss


def train_loop(_print, cfg, model, train_loader, criterion, valid_loader, valid_criterion, optimizer, scheduler, start_epoch, best_metric):
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        _print(f"\nEpoch {epoch + 1}")

        losses = AverageMeter()
        model.train()
        tbar = tqdm(train_loader)

        for i, (image, target) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()
            bsize, seq_len, c, h, w = image.size()
            image = image.view(bsize * seq_len, c, h, w)
            target = target.view(-1, target.size(-1))

            # calculate loss
            if cfg.DATA.CUTMIX:
                mixed_image, target, mixed_target, lamb = cutmix_data(image, target,
                    cfg.DATA.CM_ALPHA)
            elif cfg.DATA.MIXUP:
                mixed_image, target, mixed_target, lamb = mixup_data(image, target,
                    cfg.DATA.CM_ALPHA)
            output = model(mixed_image, seq_len)
            loss = mixup_criterion(criterion, output,
                target, mixed_target, lamb)

            # gradient accumulation
            loss = loss / cfg.OPT.GD_STEPS

            if cfg.SYSTEM.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (i + 1) % cfg.OPT.GD_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # record loss
            losses.update(loss.item() * cfg.OPT.GD_STEPS, image.size(0))
            tbar.set_description("Train loss: %.5f, learning rate: %.6f" % (losses.avg, optimizer.param_groups[-1]['lr']))

        _print("Train loss: %.5f, learning rate: %.6f" % (losses.avg, optimizer.param_groups[-1]['lr']))

        loss = valid_model(_print, cfg, model, valid_loader, valid_criterion)
        is_best = loss < best_metric
        best_metric = min(loss, best_metric)

        save_checkpoint({
            "epoch": epoch + 1,
            "arch": cfg.EXP,
            "state_dict": model.state_dict(),
            "best_metric": best_metric,
            "optimizer": optimizer.state_dict(),
        }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}.pth")


def main(args, cfg):
    # Set logger
    logging = setup_logger(args.mode, cfg.DIRS.LOGS, 0, filename=f"{cfg.EXP}.txt")

    # Declare variables
    start_epoch = 0
    best_metric = 10.

    # Create model
    model = build_model(cfg)

    # Define Loss and Optimizer
    train_criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(cfg.LOSS.WEIGHTS))
    valid_criterion = WeightedBCEWithLogitsLoss(class_weights=torch.tensor(cfg.LOSS.WEIGHTS), reduction='none')
    optimizer = make_optimizer(cfg, model)

    # CUDA & Mixed Precision
    if cfg.SYSTEM.CUDA:
        model = model.cuda()
        train_criterion = train_criterion.cuda()

    if cfg.SYSTEM.FP16:
        model, optimizer = amp.initialize(models=model, optimizers=optimizer,
                                          opt_level=cfg.SYSTEM.OPT_L,
                                          keep_batchnorm_fp32=(True if cfg.SYSTEM.OPT_L == "O2" else None))

    # Load checkpoint
    if args.load != "":
        if os.path.isfile(args.load):
            logging.info(f"=> loading checkpoint {args.load}")
            ckpt = torch.load(args.load, "cpu")
            model.load_state_dict(ckpt.pop('state_dict'))
            if not args.finetune:
                logging.info("resuming optimizer ...")
                optimizer.load_state_dict(ckpt.pop('optimizer'))
                start_epoch, best_metric = ckpt['epoch'], ckpt['best_metric']
            logging.info(f"=> loaded checkpoint '{args.load}' (epoch {ckpt['epoch']}, best_metric: {ckpt['best_metric']})")
        else:
            logging.info(f"=> no checkpoint found at '{args.load}'")

    if cfg.SYSTEM.MULTI_GPU:
        model = nn.DataParallel(model)

    if args.test_qure:
        test_ds = Qure500DS(cfg, "/home/nhannt/qure/JPEG/")
        test_loader = DataLoader(test_ds, 1, pin_memory=False, shuffle=False,
                                drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
        test_qure_dataset(logging.info, cfg, model, test_loader)
    else:
        DataSet = RSNAHemorrhageDS3d
        train_ds = DataSet(cfg, mode="train")
        valid_ds = DataSet(cfg, mode="valid")
        test_ds = DataSet(cfg, mode="test")
        if cfg.DEBUG:
            train_ds = Subset(train_ds, np.random.choice(np.arange(len(train_ds)), 50))
            valid_ds = Subset(valid_ds, np.random.choice(np.arange(len(valid_ds)), 20))

        train_loader = DataLoader(train_ds, cfg.TRAIN.BATCH_SIZE,
                                pin_memory=False, shuffle=True,
                                drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
        valid_loader = DataLoader(valid_ds, 1,
                                pin_memory=False, shuffle=False,
                                drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
        test_loader = DataLoader(test_ds, 1, pin_memory=False, shuffle=False,
                                drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)

        scheduler = make_lr_scheduler(cfg, optimizer, train_loader)
        if args.mode == "train":
            train_loop(logging.info, cfg, model, \
                    train_loader, train_criterion, valid_loader, valid_criterion, \
                    optimizer, scheduler, start_epoch, best_metric)
        elif args.mode == "valid":
            valid_model(logging.info, cfg, model, valid_loader, valid_criterion)
        else:
            submission = test_model(logging.info, cfg, model, test_loader)
            sub_fpath = os.path.join(cfg.DIRS.OUTPUTS, f"{cfg.EXP}.csv")
            submission.to_csv(sub_fpath, index=False)
            create_submission(submission, sub_fpath)


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()

    if args.config != "":
        cfg.merge_from_file(args.config)
    if args.debug:
        opts = ["DEBUG", True, "TRAIN.EPOCHS", 2]
        cfg.merge_from_list(opts)
    cfg.freeze()
    # make dirs
    for _dir in ["WEIGHTS", "OUTPUTS", "LOGS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.mkdir(cfg.DIRS[_dir])
    # seed, run
    setup_determinism(cfg.SYSTEM.SEED)
    main(args, cfg)