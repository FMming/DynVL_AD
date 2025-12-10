import os
# Early suppression of noisy third-party warnings during import
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"^CLIP\.tokenizer$")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"^kornia\.feature\.lightglue$")
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from dataset.medical_zero import MedTestDataset, MedTrainDatasetFlat
from CLIP.clip import create_model
from CLIP.tokenizer import tokenize
from CLIP.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import precision_recall_curve
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, encode_text_with_dynamic_prompt, encode_text_with_prompt_ensemble
from prompt import REAL_NAME



use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}
CLASS_INDEX_INV = {3:'Brain', 2:'Liver', 1:'Retina_RESC', -1:'Retina_OCT2017', -2:'Chest', -3:'Histopathology'}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--obj', type=str, default='Retina_RESC')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic','ensemble'], default='dynamic', help='Prompt type: dynamic modality-task or ensemble photo templates')
    parser.add_argument('--save_path', type=str, default='./ckpt/zero-shot/', help='Directory to save zero-shot checkpoints')
    parser.add_argument('--ckpt_tag', type=str, default='', help='Optional tag appended to checkpoint filename to avoid overwrite')
    # LARA attention switch
    parser.add_argument('--enable_lara', dest='enable_lara', action='store_true', help='Enable LARA attention for segmentation tokens')
    parser.add_argument('--disable_lara', dest='enable_lara', action='store_false', help='Disable LARA attention for segmentation tokens')
    parser.set_defaults(enable_lara=True)
    # VL distillation switches
    parser.add_argument('--enable_vl_distill', action='store_true', help='Enable vision-language distillation on AC (CLS vs text)')
    parser.add_argument('--vl_beta', type=float, default=0.3, help='Weight β for VL distillation loss')
    parser.add_argument('--enable_vl_dyn_coop', action='store_true', help='Use dynamic prompt features for VL teacher (override static cache)')
    # Visualization/logging removed: keep minimal training/test interface
    args = parser.parse_args()

    setup_seed(args.seed)

    # pilot alpha search removed

    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    # Prepare static text feature cache for VL distill if enabled
    if args.enable_vl_distill:
        from utils import precompute_text_feature_cache_default
        precompute_text_feature_cache_default(clip_model, device)

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list, enable_lara=args.enable_lara).to(device)
    model.eval()

    for name, param in model.named_parameters():
        param.requires_grad = True

    # self-distillation removed: do not freeze any adapters


    # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))


    # load dataset and loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = MedTrainDatasetFlat(args.data_path, args.obj, args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()


    text_feature_list = [0]
    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        for i in [1,2,3,-3,-2,-1]:
            obj_i = CLASS_INDEX_INV[i]
            task_i = 'AS' if i > 0 else 'AC'
            if args.prompt_mode == 'dynamic':
                text_feature = encode_text_with_dynamic_prompt(clip_model, obj_i, task_i, device)
            else:
                text_feature = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[obj_i], device)
            text_feature_list.append(text_feature)
        # VL teacher features per class (default from static cache; optionally dynamic override)
        if args.enable_vl_distill:
            from utils import load_cached_text_features
            text_feature_list_vl = [0]
            for i in [1,2,3,-3,-2,-1]:
                obj_i = CLASS_INDEX_INV[i]
                tf = load_cached_text_features(obj_i, 'AC', device, model=clip_model)
                text_feature_list_vl.append(tf)
            if args.enable_vl_dyn_coop:
                text_feature_list_vl = text_feature_list

    save_score = 0.0

    for epoch in range(args.epoch):
        print('epoch', epoch, ':')

        loss_list = []
        idx = 0
        for (image, image_label, mask, seg_idx) in tqdm(train_loader):
            if idx % (len(train_loader) // 5) == 0:
                score = test(args, model, test_loader, text_feature_list[CLASS_INDEX[args.obj]])
                if score >= save_score:
                    save_score = score
                    # ensure directory exists and honor --save_path
                    os.makedirs(args.save_path, exist_ok=True)
                    tag_suffix = ("_" + args.ckpt_tag) if getattr(args, 'ckpt_tag', '') else ''
                    ckp_path = os.path.join(args.save_path, f'{args.obj}{tag_suffix}.pth')
                    torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                                'det_adapters': model.det_adapters.state_dict()}, 
                                ckp_path)
                    print(f'best epoch found: epoch {epoch} batch {idx}')
                print('\n')
            idx += 1

            image = image.to(device)
            image_label = image_label.to(device)
            mask = mask.to(device)
            seg_idx = seg_idx.to(device)

            with torch.cuda.amp.autocast():
                _, seg_tokens_raw, det_tokens_raw = model(image)
                seg_patch_tokens = [p[:, 1:, :] for p in seg_tokens_raw]
                det_patch_tokens = [p[:, 1:, :] for p in det_tokens_raw]

                # 构造每样本文本特征 [B, C, 2]
                text_batch = torch.stack([text_feature_list[int(si.item())] for si in seg_idx], dim=0).to(device)
                # VL teacher batch
                if args.enable_vl_distill:
                    text_batch_vl = torch.stack([text_feature_list_vl[int(si.item())] for si in seg_idx], dim=0).to(device)

                # image-level 检测损失（向量化）
                det_loss = 0
                for layer in range(len(det_patch_tokens)):
                    p = det_patch_tokens[layer]
                    p = p / p.norm(dim=-1, keepdim=True)
                    am = torch.einsum('blc,bcq->blq', 100.0 * p, text_batch)
                    am = torch.softmax(am, dim=-1)[:, :, 1]
                    anomaly_score = am.mean(dim=1)
                    det_loss = det_loss + loss_bce(anomaly_score, image_label)

                # self-distillation removed

                # VL distillation (AC only): β * sum_{l∈{12,18,24}} [1 - cos(p_cls_l, text_ci)]
                vl_loss = torch.tensor(0.0, device=device)
                if args.enable_vl_distill:
                    B = image.shape[0]
                    # select F_text_ci per sample from text_batch_vl using image_label
                    idx_sel = image_label.long().view(B, 1, 1).expand(B, text_batch_vl.shape[1], 1)
                    text_ci = text_batch_vl.gather(2, idx_sel).squeeze(2)  # [B, C]
                    # apply on chosen layers
                    vl_layers = [l for l in [12, 18, 24] if l in args.features_list]
                    for vl in vl_layers:
                        s_idx = args.features_list.index(vl)
                        p_cls = det_tokens_raw[s_idx][:, 0, :]
                        sim = F.cosine_similarity(p_cls, text_ci, dim=1)
                        vl_loss = vl_loss + (1.0 - sim).mean()
                    vl_loss = args.vl_beta * vl_loss

                # pixel-level 分割损失（仅对 seg_idx>0 的样本）
                idx_seg = (seg_idx > 0)
                if idx_seg.any():
                    seg_loss = 0
                    mask_bin = mask.clone()
                    mask_bin[mask_bin > 0.5], mask_bin[mask_bin <= 0.5] = 1, 0
                    mask_sel = mask_bin[idx_seg]
                    for layer in range(len(seg_patch_tokens)):
                        p = seg_patch_tokens[layer][idx_seg]
                        p = p / p.norm(dim=-1, keepdim=True)
                        text_sel = text_batch[idx_seg]
                        am = torch.einsum('blc,bcq->blq', 100.0 * p, text_sel)
                        B_cur, L, C2 = am.shape
                        H = int(np.sqrt(L))
                        am = F.interpolate(am.permute(0, 2, 1).view(B_cur, 2, H, H), size=args.img_size, mode='bilinear', align_corners=True)
                        am = torch.softmax(am, dim=1)
                        seg_loss = seg_loss + loss_focal(am, mask_sel)
                        seg_loss = seg_loss + loss_dice(am[:, 1, :, :], mask_sel)

                    # self-distillation removed from loss
                    loss = seg_loss + det_loss + vl_loss
                    loss.requires_grad_(True)
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    loss.backward()
                    seg_optimizer.step()
                    det_optimizer.step()
                else:
                    loss = det_loss + vl_loss
                    loss.requires_grad_(True)
                    det_optimizer.zero_grad()
                    loss.backward()
                    det_optimizer.step()

                # visualization/logging removed

            loss_list.append(loss.item())

        print(f'epoch {epoch}: loss {np.mean(loss_list):.4f}')

        train_dataset.shuffle_dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        # logs
        print("Loss: ", np.mean(loss_list))
        



def test(args, seg_model, test_loader, text_features, return_metrics=False):
    gt_list = []
    gt_mask_list = []
    image_scores = []
    segment_scores = []
    
    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, ori_seg_patch_tokens, ori_det_patch_tokens = seg_model(image)
            ori_seg_patch_tokens = [p[:, 1:, :] for p in ori_seg_patch_tokens]
            ori_det_patch_tokens = [p[:, 1:, :] for p in ori_det_patch_tokens]

            # image-level（批量）
            anomaly_score = 0
            for layer in range(len(ori_det_patch_tokens)):
                p = ori_det_patch_tokens[layer]
                p = p / p.norm(dim=-1, keepdim=True)
                am = (100.0 * p @ text_features)
                am = torch.softmax(am, dim=-1)[:, :, 1]  # [B, L]
                anomaly_score = anomaly_score + am.mean(dim=1)  # [B]
            for b in range(image.shape[0]):
                image_scores.append(anomaly_score[b].cpu().item())

            # pixel-level（批量）
            anomaly_maps_layers = []
            for layer in range(len(ori_seg_patch_tokens)):
                p = ori_seg_patch_tokens[layer]
                p = p / p.norm(dim=-1, keepdim=True)
                am = (100.0 * p @ text_features)  # [B, L, 2]
                B_cur, L, C2 = am.shape
                H = int(np.sqrt(L))
                am = F.interpolate(am.permute(0, 2, 1).view(B_cur, 2, H, H), size=args.img_size, mode='bilinear', align_corners=True)
                am = torch.softmax(am, dim=1)[:, 1, :, :]  # [B, H, H]
                anomaly_maps_layers.append(am.cpu().numpy())

            anomaly_maps_layers = np.array(anomaly_maps_layers)  # [num_layers, B, H, H]
            final_seg_map = np.sum(anomaly_maps_layers, axis=0)  # [B, H, H]
            for b in range(final_seg_map.shape[0]):
                segment_scores.append(final_seg_map[b])

            gt_mask_list.extend(mask.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
        
        

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)

    segment_scores = np.array(segment_scores)
    image_scores = np.array(image_scores)

    segment_scores = (segment_scores - segment_scores.min()) / (segment_scores.max() - segment_scores.min())
    image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min())

    img_roc_auc_det = roc_auc_score(gt_list, image_scores)
    print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

    if CLASS_INDEX[args.obj] > 0:
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')
        if return_metrics:
            return {'seg_pAUC': float(seg_roc_auc), 'img_AUC': float(img_roc_auc_det)}
        return seg_roc_auc + img_roc_auc_det
    else:
        if return_metrics:
            return {'seg_pAUC': None, 'img_AUC': float(img_roc_auc_det)}
        return img_roc_auc_det


# pilot alpha sweep removed


if __name__ == '__main__':
    main()


