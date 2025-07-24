# utils/utils_fit.py

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

try:
    from nets.lovasz_losses import lovasz_softmax
    LOVASZ_AVAILABLE = True
except ImportError:
    # This warning is useful if lovasz is intended but not found
    # print("Warning: lovasz_softmax not found in nets.lovasz_losses, Lovasz Loss will be unavailable.")
    lovasz_softmax = None
    LOVASZ_AVAILABLE = False

# 确保这里的导入指向你更新后的 segformer_training.py
from nets.segwater_training import (CE_Loss, Dice_loss, Focal_Loss, ConLoss)
from utils.utils import get_lr
from utils.utils_metrics import f_score


def get_model_outputs(model_output_raw, num_expected_outputs=5):
    if isinstance(model_output_raw, list) and len(model_output_raw) > 0: # DataParallel
        output_item = model_output_raw[0]
        if isinstance(output_item, tuple) and len(output_item) == num_expected_outputs:
            return output_item
        elif not isinstance(output_item, tuple) and num_expected_outputs == 1:
            return (output_item,)
        else:
            type_info = type(output_item); len_info = len(output_item) if isinstance(output_item, tuple) else 'NotTuple'
            raise TypeError(f"DP output item unexp: {type_info}, len: {len_info}. Exp {num_expected_outputs} elem.")
    elif isinstance(model_output_raw, tuple) and len(model_output_raw) == num_expected_outputs:
        return model_output_raw
    elif not isinstance(model_output_raw, tuple) and num_expected_outputs == 1: # Single tensor output
        return (model_output_raw,)
    else:
        type_info = type(model_output_raw); len_info = len(model_output_raw) if isinstance(model_output_raw, tuple) else 'NotTuple'
        raise TypeError(f"Model output type unexp: {type_info}, len: {len_info}. Exp {num_expected_outputs} elem.")


def calculate_loss_and_fscore(current_seg_outputs, current_r1, current_r3,
                              current_aux_pred_c2, current_aux_pred_c3, # 这些现在是可选的辅助预测
                              current_pngs_idx_map, current_con_1_target, current_con_3_target,
                              current_seg_labels_one_hot, weights, num_classes, focal_loss_flag, dice_loss_flag,
                              lovasz_alpha,
                              using_deep_supervision_flag,
                              aux_loss_weight,
                              is_training=True,
                              dice_smooth=1e-5,
                              local_rank=0,
                              epoch=-1, iteration=-1):
    device = torch.device("cpu")
    # Determine device from the first available tensor, crucial for tensor creation
    if current_seg_outputs is not None and isinstance(current_seg_outputs, torch.Tensor): device = current_seg_outputs.device
    elif current_r1 is not None and isinstance(current_r1, torch.Tensor): device = current_r1.device
    elif current_aux_pred_c2 is not None and isinstance(current_aux_pred_c2, torch.Tensor): device = current_aux_pred_c2.device
    # Add more checks if necessary, or ensure at least one tensor is always passed

    base_main_loss = torch.tensor(0.0, device=device)
    if current_seg_outputs is not None and current_pngs_idx_map is not None and weights is not None:
        if focal_loss_flag:
            # current_pngs_idx_map 形状 (B, H, W) 传递给 Focal_Loss 的 target 参数
            base_main_loss = Focal_Loss(inputs=current_seg_outputs, target=current_pngs_idx_map, cls_weights=weights, num_classes=num_classes)
        else:
            # current_pngs_idx_map 形状 (B, H, W) 传递给 CE_Loss 的 target 参数
            base_main_loss = CE_Loss(inputs=current_seg_outputs, target=current_pngs_idx_map, cls_weights=weights, num_classes=num_classes)

    lovasz_loss_val = torch.tensor(0.0, device=device)
    if LOVASZ_AVAILABLE and lovasz_alpha > 0 and current_seg_outputs is not None and current_pngs_idx_map is not None:
        try:
            seg_probas = F.softmax(current_seg_outputs, dim=1)
            # lovasz_softmax 也期望 labels 是 (B, H, W) 形状的类别索引
            lovasz_loss_val = lovasz_softmax(probas=seg_probas, labels=current_pngs_idx_map, classes='present', per_image=False, ignore=None)
        except Exception as e:
            if local_rank == 0: print(f"Lovasz loss error (ep {epoch}, it {iteration}): {e}")
    main_loss = (1 - lovasz_alpha) * base_main_loss + lovasz_alpha * lovasz_loss_val
    current_total_loss = main_loss

    dice = torch.tensor(0.0, device=device)
    if dice_loss_flag and current_seg_outputs is not None and current_seg_labels_one_hot is not None:
        # current_seg_labels_one_hot 形状 (B, H, W, num_classes+1) 传递给 Dice_loss 的 target 参数
        dice = Dice_loss(inputs=current_seg_outputs, target=current_seg_labels_one_hot, smooth=dice_smooth)
        current_total_loss += dice

    c1_loss_val, c3_loss_val = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    if current_r1 is not None and current_con_1_target is not None:
        # current_con_1_target 形状 (B, H, W, 9) 传递给 ConLoss 的 gt_con 参数
        target_size_con1 = current_r1.shape[-2:]
        # r1_upsampled = F.interpolate(current_r1, size=target_size_con1, mode='bilinear', align_corners=False) if current_r1.shape[-2:] != target_size_con1 else current_r1
        # ConLoss will handle target permutation, ensure pred_con matches upsampled r1
        c1_loss_val = ConLoss(current_r1, current_con_1_target) # No need to upsample r1 if ConLoss handles shape

    if current_r3 is not None and current_con_3_target is not None:
        # current_con_3_target 形状 (B, H, W, 9) 传递给 ConLoss 的 gt_con 参数
        target_size_con3 = current_r3.shape[-2:]
        # r3_upsampled = F.interpolate(current_r3, size=target_size_con3, mode='bilinear', align_corners=False) if current_r3.shape[-2:] != target_size_con3 else current_r3
        c3_loss_val = ConLoss(current_r3, current_con_3_target) # No need to upsample r3 if ConLoss handles shape
    lad = 0.4 if is_training else 0.2
    con_total_loss = lad * (0.6 * c3_loss_val + 0.4 * c1_loss_val)
    current_total_loss += con_total_loss

    loss_aux_c2, loss_aux_c3, total_aux_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    if using_deep_supervision_flag:
        # 这里的 deep supervision 期望 model 返回 current_aux_pred_c2/c3
        # 但 SegFormer 模型并没有返回这些。
        # 因此，这些辅助损失会是 0.0，除非你修改模型使其返回这些预测。
        if current_aux_pred_c2 is not None and current_pngs_idx_map is not None and weights is not None:
            gt_down = F.interpolate(current_pngs_idx_map.unsqueeze(1).float(), size=current_aux_pred_c2.shape[2:], mode='nearest').squeeze(1).long()
            loss_aux_c2 = CE_Loss(inputs=current_aux_pred_c2, target=gt_down, cls_weights=weights, num_classes=num_classes)
        if current_aux_pred_c3 is not None and current_pngs_idx_map is not None and weights is not None:
            gt_down = F.interpolate(current_pngs_idx_map.unsqueeze(1).float(), size=current_aux_pred_c3.shape[2:], mode='nearest').squeeze(1).long()
            loss_aux_c3 = CE_Loss(inputs=current_aux_pred_c3, target=gt_down, cls_weights=weights, num_classes=num_classes)
        total_aux_loss = aux_loss_weight * (loss_aux_c2 + loss_aux_c3)
        current_total_loss += total_aux_loss

    print_loss_debug = (torch.isnan(current_total_loss).any() or torch.isinf(current_total_loss).any()) or \
                       (is_training and iteration > 0 and iteration % 200 == 0)
    if local_rank == 0 and print_loss_debug:
        print(f"\n--- Loss (Ep {epoch}, It {iteration}, Train: {is_training}) ---")
        print(f"  BaseMain: {base_main_loss.item():.4f}, Lovasz: {lovasz_loss_val.item():.4f} (a={lovasz_alpha})")
        print(f"  Dice: {dice.item() if dice_loss_flag else 'N/A':<7}, ConC1: {c1_loss_val.item():.4f}, ConC3: {c3_loss_val.item():.4f} (TotalCon: {con_total_loss.item():.4f}, lad={lad})")
        if using_deep_supervision_flag: print(f"  AuxC2: {loss_aux_c2.item():.4f}, AuxC3: {loss_aux_c3.item():.4f} (TotalAux: {total_aux_loss.item():.4f}, w={aux_loss_weight})")
        print(f"  ==> TOTAL: {current_total_loss.item():.4f}")
        if torch.isnan(current_total_loss).any() or torch.isinf(current_total_loss).any(): print("  !!!!!! CRITICAL: NaN/Inf loss !!!!!!")
        print("--- End Loss ---")

    f_score_val = torch.tensor(0.0, device=device)
    if current_seg_outputs is not None and current_pngs_idx_map is not None:
        with torch.no_grad():
            pred_indices = torch.argmax(current_seg_outputs, dim=1)
            try: f_score_val = f_score(pred_indices, current_pngs_idx_map)
            except Exception as e:
                if local_rank == 0: print(f"F-score error (ep {epoch}, it {iteration}): {e}")
    return current_total_loss, f_score_val


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda,
                  dice_loss_flag, focal_loss_flag, cls_weights, num_classes, fp16, scaler, save_period, save_dir,
                  lovasz_alpha, using_deep_supervision_flag, aux_loss_weight,
                  local_rank=0, dice_smooth_val=1e-5, gradient_clip_val=None, 
                  num_expected_model_outputs=3): # Adjust num_expected_model_outputs to 3 for SegFormer

    total_loss_accum, total_f_score_accum, val_loss_accum, val_f_score_accum = 0.0, 0.0, 0.0, 0.0 # Initialize as float
    batches_processed_train, batches_processed_val = 0, 0
    weights_device = torch.device(f"cuda:{local_rank}" if cuda else "cpu")
    weights_tensor_prototype = torch.from_numpy(cls_weights).to(weights_device, non_blocking=True, dtype=torch.half if fp16 and cuda else torch.float)

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: break
        try:
            # 解包批次数据，确保顺序和数量正确
            imgs, pngs_idx_map, con_1_target, con_3_target, seg_labels_one_hot_target = batch
        except ValueError as e:
            if local_rank == 0: print(f"Error unpacking train batch iter {iteration}: {e}. Skip.")
            continue

        if cuda: # Move data to device
            imgs = imgs.to(weights_device, non_blocking=True)
            pngs_idx_map = pngs_idx_map.to(weights_device, non_blocking=True)
            con_1_target = con_1_target.to(weights_device, non_blocking=True) if con_1_target is not None else None
            con_3_target = con_3_target.to(weights_device, non_blocking=True) if con_3_target is not None else None
            seg_labels_one_hot_target = seg_labels_one_hot_target.to(weights_device, non_blocking=True)
        
        weights_tensor = weights_tensor_prototype # Use pre-moved, correctly-typed tensor
        optimizer.zero_grad(set_to_none=True) # More memory efficient

        current_iter_loss = torch.tensor(0.0, device=weights_device) 
        current_iter_f_score = torch.tensor(0.0, device=weights_device)

        if not fp16 or not cuda: # CPU or FP32 on GPU
            raw_outputs = model_train(imgs)
            try:
                outputs_tuple = get_model_outputs(raw_outputs, num_expected_model_outputs)
                # outputs_tuple 包含 (seg_pred, con0_pred, con1_pred)
                current_seg_outputs = outputs_tuple[0]
                current_r1 = outputs_tuple[1]
                current_r3 = outputs_tuple[2]
                # current_aux_pred_c2 和 current_aux_pred_c3 模型并未返回，显式设置为 None
                current_aux_pred_c2 = None
                current_aux_pred_c3 = None
            except TypeError as e:
                if local_rank == 0: print(f"Error get_model_outputs (train, no_fp16, iter {iteration}): {e}"); continue
            
            # 使用命名参数，确保参数正确映射
            current_iter_loss, current_iter_f_score = calculate_loss_and_fscore(
                current_seg_outputs=current_seg_outputs,
                current_r1=current_r1,
                current_r3=current_r3,
                current_aux_pred_c2=current_aux_pred_c2,
                current_aux_pred_c3=current_aux_pred_c3,
                current_pngs_idx_map=pngs_idx_map,
                current_con_1_target=con_1_target,
                current_con_3_target=con_3_target,
                current_seg_labels_one_hot=seg_labels_one_hot_target,
                weights=weights_tensor,
                num_classes=num_classes,
                focal_loss_flag=focal_loss_flag,
                dice_loss_flag=dice_loss_flag,
                lovasz_alpha=lovasz_alpha,
                using_deep_supervision_flag=using_deep_supervision_flag,
                aux_loss_weight=aux_loss_weight,
                is_training=True,
                dice_smooth=dice_smooth_val,
                local_rank=local_rank,
                epoch=epoch,
                iteration=iteration
            )
            
            if not (torch.isnan(current_iter_loss).any() or torch.isinf(current_iter_loss).any()):
                current_iter_loss.backward()
                if gradient_clip_val: torch.nn.utils.clip_grad_norm_(model_train.parameters(), gradient_clip_val)
                optimizer.step()
                total_loss_accum += current_iter_loss.item(); total_f_score_accum += current_iter_f_score.item(); batches_processed_train += 1
            elif local_rank == 0: print(f"CRIT: NaN/Inf loss TRAIN (no_fp16) ep {epoch} it {iteration}. Skip back/step.")
        
        else: # FP16 on GPU
            from torch.cuda.amp import autocast
            with autocast():
                raw_outputs = model_train(imgs)
                try: 
                    outputs_tuple = get_model_outputs(raw_outputs, num_expected_model_outputs)
                    # outputs_tuple 包含 (seg_pred, con0_pred, con1_pred)
                    current_seg_outputs = outputs_tuple[0]
                    current_r1 = outputs_tuple[1]
                    current_r3 = outputs_tuple[2]
                    current_aux_pred_c2 = None # 显式设置为 None
                    current_aux_pred_c3 = None # 显式设置为 None
                except TypeError as e:
                    if local_rank == 0: print(f"Error get_model_outputs (train, fp16, iter {iteration}): {e}"); continue
                
                # 使用命名参数，确保参数正确映射
                current_iter_loss, current_iter_f_score = calculate_loss_and_fscore(
                    current_seg_outputs=current_seg_outputs,
                    current_r1=current_r1,
                    current_r3=current_r3,
                    current_aux_pred_c2=current_aux_pred_c2,
                    current_aux_pred_c3=current_aux_pred_c3,
                    current_pngs_idx_map=pngs_idx_map,
                    current_con_1_target=con_1_target,
                    current_con_3_target=con_3_target,
                    current_seg_labels_one_hot=seg_labels_one_hot_target,
                    weights=weights_tensor,
                    num_classes=num_classes,
                    focal_loss_flag=focal_loss_flag,
                    dice_loss_flag=dice_loss_flag,
                    lovasz_alpha=lovasz_alpha,
                    using_deep_supervision_flag=using_deep_supervision_flag,
                    aux_loss_weight=aux_loss_weight,
                    is_training=True,
                    dice_smooth=dice_smooth_val,
                    local_rank=local_rank,
                    epoch=epoch,
                    iteration=iteration
                )
            
            if not (torch.isnan(current_iter_loss).any() or torch.isinf(current_iter_loss).any()):
                scaler.scale(current_iter_loss).backward()
                if gradient_clip_val: 
                    scaler.unscale_(optimizer) # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(model_train.parameters(), gradient_clip_val)
                scaler.step(optimizer); scaler.update()
                total_loss_accum += current_iter_loss.item(); total_f_score_accum += current_iter_f_score.item(); batches_processed_train += 1
            elif local_rank == 0: 
                print(f"CRIT: NaN/Inf loss FP16 TRAIN ep {epoch} it {iteration}. Skip back/step.")
                optimizer.zero_grad(set_to_none=True) # Clear gradients as optimizer.step was skipped

        if local_rank == 0:
            avg_loss = total_loss_accum / batches_processed_train if batches_processed_train > 0 else float('nan')
            avg_fscore = total_f_score_accum / batches_processed_train if batches_processed_train > 0 else float('nan')
            pbar.set_postfix(**{'loss': f"{avg_loss:.4f}", 'f_sc': f"{avg_fscore:.4f}", 'lr': f"{get_lr(optimizer):.1e}"})
            pbar.update(1)

    if local_rank == 0:
        pbar.close(); print('Finish Train\nStart Validation')
        pbar_val = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch} (Val)', postfix=dict, mininterval=0.3)

    model_train.eval() # Set model to evaluation mode
    for iteration_val, batch_val in enumerate(gen_val):
        if iteration_val >= epoch_step_val: break
        try:
            imgs_val, pngs_idx_map_val, con_1_target_val, con_3_target_val, seg_labels_one_hot_target_val = batch_val
        except ValueError as e:
            if local_rank == 0: print(f"Error unpacking VAL batch iter {iteration_val}: {e}. Skip.")
            continue
        
        if cuda: # Move validation data to device
            imgs_val = imgs_val.to(weights_device,non_blocking=True)
            pngs_idx_map_val = pngs_idx_map_val.to(weights_device,non_blocking=True)
            con_1_target_val = con_1_target_val.to(weights_device,non_blocking=True) if con_1_target_val is not None else None
            con_3_target_val = con_3_target_val.to(weights_device,non_blocking=True) if con_3_target_val is not None else None
            seg_labels_one_hot_target_val = seg_labels_one_hot_target_val.to(weights_device,non_blocking=True)
        weights_tensor_val = weights_tensor_prototype

        with torch.no_grad(): # Ensure no gradients are computed during validation
            seg_outputs_val = None # Initialize
            if not fp16 or not cuda:
                # In eval mode, SegFormer returns only the main segmentation output (seg_pred)
                raw_outputs_val = model_train(imgs_val) 
                seg_outputs_val = raw_outputs_val 
            else: # FP16 on GPU
                from torch.cuda.amp import autocast
                with autocast():
                    raw_outputs_val = model_train(imgs_val)
                    seg_outputs_val = raw_outputs_val
            
            # 验证集计算损失时，只计算主分割损失和Dice/Focal/Lovasz，不计算连接性损失和辅助损失。
            # 这是因为 model_train.eval() 模式下，SegFormer 只返回 seg_pred。
            current_val_iter_loss, current_val_iter_f_score = calculate_loss_and_fscore(
                current_seg_outputs=seg_outputs_val,
                current_r1=None, # Model in eval mode only returns seg_pred
                current_r3=None, # Model in eval mode only returns seg_pred
                current_aux_pred_c2=None, # Auxiliary predictions are not returned in eval mode
                current_aux_pred_c3=None, # Auxiliary predictions are not returned in eval mode
                current_pngs_idx_map=pngs_idx_map_val, 
                current_con_1_target=None, # Connection targets not used for validation loss
                current_con_3_target=None, # Connection targets not used for validation loss
                current_seg_labels_one_hot=seg_labels_one_hot_target_val, 
                weights=weights_tensor_val, num_classes=num_classes, 
                focal_loss_flag=focal_loss_flag, dice_loss_flag=dice_loss_flag, lovasz_alpha=lovasz_alpha, 
                using_deep_supervision_flag=False, # Deep supervision typically off for validation loss metric
                aux_loss_weight=aux_loss_weight,   
                is_training=False, dice_smooth=dice_smooth_val,
                local_rank=local_rank, epoch=epoch, iteration=iteration_val)

            if not (torch.isnan(current_val_iter_loss).any() or torch.isinf(current_val_iter_loss).any()):
                val_loss_accum += current_val_iter_loss.item(); val_f_score_accum += current_val_iter_f_score.item(); batches_processed_val += 1
            elif local_rank == 0: print(f"WARN: NaN/Inf loss VAL ep {epoch} it {iteration_val}.")
        
        if local_rank == 0:
            avg_val_loss = val_loss_accum / batches_processed_val if batches_processed_val > 0 else float('nan')
            avg_val_fscore = val_f_score_accum / batches_processed_val if batches_processed_val > 0 else float('nan')
            pbar_val.set_postfix(**{'val_loss': f"{avg_val_loss:.4f}", 'f_score': f"{avg_val_fscore:.4f}"})
            pbar_val.update(1)

    if local_rank == 0:
        pbar_val.close(); print('Finish Validation')
        avg_train_loss = total_loss_accum / batches_processed_train if batches_processed_train > 0 else float('nan')
        avg_val_loss = val_loss_accum / batches_processed_val if batches_processed_val > 0 else float('nan')
        
        if loss_history: loss_history.append_loss(epoch + 1, avg_train_loss, avg_val_loss)
        
        model_to_eval_or_save = model.module if isinstance(model_train, torch.nn.DataParallel) else model
        if eval_callback: eval_callback.on_epoch_end(epoch + 1, model_to_eval_or_save)
            
        print(f'Epoch:{epoch+1}/{Epoch} Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}')
        
        train_loss_fn_str = f"{avg_train_loss:.3f}" if not np.isnan(avg_train_loss) else "nan"
        val_loss_fn_str = f"{avg_val_loss:.3f}" if not np.isnan(avg_val_loss) else "nan"

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            save_path = os.path.join(save_dir, f'ep{epoch+1:03d}-l{train_loss_fn_str}-vl{val_loss_fn_str}.pth')
            torch.save(model_to_eval_or_save.state_dict(), save_path)
            print(f"Saved model to {save_path}")

        if not np.isnan(avg_val_loss) and loss_history:
            valid_past_val = [l for l in loss_history.val_loss if not np.isnan(l)]
            if not valid_past_val or avg_val_loss <= min(valid_past_val, default=float('inf')):
                best_save_path = os.path.join(save_dir, "best_epoch_weights.pth")
                print(f'Saving best model to {best_save_path} (Val Loss: {avg_val_loss:.4f})')
                torch.save(model_to_eval_or_save.state_dict(), best_save_path)
        
        last_save_path = os.path.join(save_dir, "last_epoch_weights.pth")
        torch.save(model_to_eval_or_save.state_dict(), last_save_path)
        print(f"Saved last model to {last_save_path}")