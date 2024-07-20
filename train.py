import os
import torch
import argparse
from tqdm import tqdm
import random
import numpy as np
from eval import eval_for_metric
from losses.get_losses import SelectLoss
from models.block.Drop import dropblock_step
from utils.dataloaders import get_loaders
from utils.common import check_dirs, init_seed, gpu_info, SaveResult, CosOneCycle, ScaleInOutput
from models.main_model import ChangeDetection, ModelEMA, ModelSWA
import torch.nn.functional as F
from torch import optim
from sklearn.cluster import KMeans
    
def downsample_label(label, new_size):
    batch_size, height, width = label.shape
    new_height, new_width = new_size

    stride_h = int(height / new_height)
    stride_w = int(width / new_width)

    downsampled_label = torch.zeros((batch_size, new_height, new_width), dtype=label.dtype, device=label.device)

    for i in range(0, height, stride_h):
        for j in range(0, width, stride_w):
            block = label[:, i:i+stride_h, j:j+stride_w]
            count_ones = block.sum(dim=[1, 2])
            count_zeros = stride_h * stride_w - count_ones

            downsampled_label[:, i//stride_h, j//stride_w] = (count_ones >= count_zeros).float()

    return downsampled_label

def train(opt):
    # init_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    gpu_info()
    save_path, best_ckp_save_path, best_ckp_file, result_save_path, every_ckp_save_path = check_dirs()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_results = SaveResult(result_save_path)
    save_results.prepare()

    train_loader, val_loader = get_loaders(opt)
    scale = ScaleInOutput(opt.input_size)

    model = ChangeDetection(opt).cuda()
    if torch.cuda.device_count() > 1:
        # model = torch.nn.DataParallel(model)
        model = torch.nn.DataParallel(model,device_ids = [0,1,2,3])
    criterion = SelectLoss(opt.loss)

    if opt.finetune:
        params = [{"params": [param for name, param in model.named_parameters()
                              if "backbone" in name], "lr": opt.learning_rate / 10},  
                  {"params": [param for name, param in model.named_parameters()
                              if "backbone" not in name], "lr": opt.learning_rate}]  
        print("Using finetune for model")
    else:
        params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=opt.learning_rate, weight_decay=0.01)
    #optimizer = torch.optim.SGD(params, lr=opt.learning_rate, weight_decay=0.5)
    if opt.pseudo_label:
        scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate/5, epochs=opt.epochs, up_rate=0)
    else:
        scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate, epochs=opt.epochs, up_rate=0)  

    best_metric = 0
    train_avg_loss = 0
    train_dis_loss = 0
    total_bs = 32
    accumulate_iter = max(round(total_bs / opt.batch_size), 1)
    print("Accumulate_iter={} batch_size={}".format(accumulate_iter, opt.batch_size))

    for epoch in range(opt.epochs):
        model.train()
        train_tbar = tqdm(train_loader)
        for i, (batch_img1, batch_img2, batch_label, batch_label2, _) in enumerate(train_tbar):
            train_tbar.set_description("epoch {}, train_loss {}".format(epoch, train_avg_loss))
            if epoch == 0 and i < 20:
                save_results.save_first_batch(batch_img1, batch_img2, batch_label, batch_label2, i)

            batch_img1 = batch_img1.float().cuda()
            batch_img2 = batch_img2.float().cuda()
            batch_label = batch_label.long().cuda()
            batch_label2 = batch_label2.long().cuda()

            batch_img1, batch_img2 = scale.scale_input((batch_img1, batch_img2)) 
            outs, diff1, diff2, diff3, diff4 = model(batch_img1, batch_img2)
            outs = scale.scale_output(outs)
            new_size = (diff1.shape[2], diff1.shape[3])
            downsampled_label = downsample_label(batch_label, new_size)

            batch_size = diff1.shape[0] 
            exp_losses = torch.zeros(batch_size, device=diff1.device)
            for i in range(batch_size):
                sample_diff1 = diff1[i]
                sample_downsampled_label = downsampled_label[i]
                # print(sample_diff1.shape,sample_downsampled_label.shape)
                mask_change = (sample_downsampled_label == 1)
                mask_no_change = (sample_downsampled_label == 0)

                l2_distance = torch.norm(sample_diff1, dim=0)
                mask_no_change_clone = mask_no_change.clone()
                # l2_distance = sample_diff1.squeeze(0)
                l2_distance_no_change = l2_distance[mask_no_change].detach().cpu().numpy()
                l2_distance_no_change = l2_distance_no_change[np.isfinite(l2_distance_no_change)]
                if l2_distance_no_change.size > 0 and epoch < 10:
                    kmeans = KMeans(n_clusters=2, init='k-means++').fit(l2_distance_no_change.reshape(-1, 1))
                    labels = kmeans.labels_
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    smaller_cluster_center_idx = np.argmin(kmeans.cluster_centers_)

                    selected_cluster = smaller_cluster_center_idx
                    mask_no_change_clone[mask_no_change] = torch.tensor(labels == selected_cluster, dtype=torch.bool, device=mask_no_change.device)

                T1, T2 = 1, 1  
                loss_change = (l2_distance[mask_change].sum()) / (mask_change.sum() + 1e-8)
                loss_no_change = (l2_distance[mask_no_change_clone].sum()) / (mask_no_change_clone.sum() + 1e-8)
                if loss_change > 0:
                    exp_loss = torch.exp(loss_no_change / T1) + 1 / (torch.exp(loss_change / T2))
                else:
                    exp_loss = torch.exp(loss_no_change / T1)
                exp_losses[i] = exp_loss

            total_loss_exp = 0.005 * exp_losses.mean()
            loss = criterion(outs, (batch_label,))
            loss = loss + total_loss_exp

            train_avg_loss = (train_avg_loss * i + loss.cpu().detach().numpy()) / (i + 1)
            train_dis_loss = (train_dis_loss * i + total_loss_exp.cpu().detach().numpy()) / (i + 1)

            loss.backward()
            if ((i+1) % accumulate_iter) == 0:
                optimizer.step()
                optimizer.zero_grad()

            del batch_img1, batch_img2, batch_label, batch_label2

        scheduler.step()
        dropblock_step(model)
        p, r, f1, miou, oa, val_avg_loss = eval_for_metric(model, val_loader, criterion, input_size=opt.input_size)

        # refer_metric = f1
        refer_metric = f1
        underscore = "_"
        if refer_metric.mean() > best_metric and refer_metric.mean() < 0.9125:
            if best_ckp_file is not None:
                os.remove(best_ckp_file)
            best_ckp_file = os.path.join(
                best_ckp_save_path,
                underscore.join([opt.backbone, opt.neck, opt.head, 'epoch',
                                 str(epoch), str(round(float(refer_metric.mean()), 5))]) + ".pt")
            torch.save(model, best_ckp_file)
            best_metric = refer_metric.mean()

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        save_results.show(p, r, f1, miou, oa, refer_metric, best_metric, train_avg_loss, val_avg_loss, lr, epoch,train_dis_loss)

def set_randomness():
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection train')
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--neck", type=str, default="fpn+aspp+fuse+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--loss", type=str, default="bce+dice")

    parser.add_argument("--pretrain", type=str,
                        default="")
    parser.add_argument("--cuda", type=str, default="1")
    parser.add_argument("--dataset-dir", type=str, default="/mnt/data_2/datasets/Change_Detection/LEVIR_cut")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--dual-label", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=True)
    parser.add_argument("--pseudo-label", type=bool, default=True)

    opt = parser.parse_args()
    #print("\n" + "-" * 30 + "OPT" + "-" * 30)
    print(opt)
    set_randomness()
    train(opt)
