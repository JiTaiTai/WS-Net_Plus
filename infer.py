import os
import cv2
import time
import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm

from models.main_model import EnsembleModel
from utils.dataloaders import get_infer_loaders
from utils.common import gpu_info, get_metrics_offline


def infer(opt):
    crop_size = gm.get_value("size")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_info()  
    
    infer_loader = get_infer_loaders(opt)

    # 加载模型
    model = EnsembleModel(opt.ckp_paths, device, input_size=opt.input_size)

    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)

    change1_save_path = os.path.join(opt.save_path, 'label1')
    change2_save_path = os.path.join(opt.save_path, 'label2')
    if os.path.isdir(change1_save_path):
        shutil.rmtree(change1_save_path)
    os.mkdir(change1_save_path)
    if os.path.isdir(change2_save_path):
        shutil.rmtree(change2_save_path)
    os.mkdir(change2_save_path)

    single_out = False
    model.eval()
    with torch.no_grad():
        infer_tbar = tqdm(infer_loader)
        for batch_img1, batch_img2, names in infer_tbar:
            infer_tbar.set_description("Infering")
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            _, _, img_h, img_w = batch_img1.shape    

           
            cd_preds = model(batch_img1, batch_img2, opt.tta)  

            if not isinstance(cd_preds, tuple):
                cd_preds = (cd_preds, cd_preds)
                single_out = True

            cd_pred1, cd_pred2 = cd_preds
            cd_pred1 = cd_pred1.data.cpu().numpy().squeeze() * 255
            cd_pred2 = cd_pred2.data.cpu().numpy().squeeze() * 255


            if opt.batch_size == 1: 
                cd_pred1 = np.expand_dims(cd_pred1, axis=0)
                cd_pred2 = np.expand_dims(cd_pred2, axis=0)

            change_mask1 = np.zeros((img_w, img_h, 3))
            change_mask2 = np.zeros((img_w, img_h, 3))
            for img_batch_id in range(cd_preds[0].shape[0]):
                name = names[img_batch_id].replace("tif", "png")
                change1_save_file_path = os.path.join(change1_save_path, name)
                change2_save_file_path = os.path.join(change2_save_path, name)
                change_mask1[:, :, 0] = cd_pred1[img_batch_id]   # blue
                change_mask2[:, :, 2] = cd_pred2[img_batch_id]   # red

                if single_out:
                    final_out = change_mask1[:crop_size, :crop_size, 0]
                    final_out = cv2.resize(final_out, (512, 512), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(change1_save_file_path, final_out)  
                    
                else:
                    cv2.imwrite(change1_save_file_path, change_mask1)
                    cv2.imwrite(change2_save_file_path, change_mask2)


import utils.GlobalManager as gm
gm._init()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection infer main')

    # 配置推理参数
    parser.add_argument("--ckp_paths", type=str,
                        default=[
                            "./runs/train/27/best_ckp/",
                        ])
    parser.add_argument("--cuda", type=str, default="7")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--tta", type=bool, default=False)
    parser.add_argument("--dataset-dir", default='/zq2/dataset/CD-dataset/whu/test', help="input path", type=str)
    parser.add_argument("--save-path", default='./infer', help="output path", type=str)

    opt = parser.parse_args()
    print(opt)
    for size in [512]:
        for inputsize in [512]:
            opt.input_size = inputsize
            gm.set_value("size", size)
            print("-"*20+str(size)+"-"*20+str(inputsize))

            infer(opt)
            get_metrics_offline(opt.dataset_dir, opt.save_path, dual_label=False)

