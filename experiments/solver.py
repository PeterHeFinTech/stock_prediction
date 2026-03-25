import torch
import torch.distributed as dist
from torch.amp import autocast
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import time
import numpy as np
import sys
import os

# Add parent directory to path to import metrics
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.metrics import (
    sharpe_ratio, portfolio_turnover, net_of_cost_sharpe,
    long_short_portfolio_returns, cross_entropy_loss, mcfadden_pseudo_r2,
    annualized_return, annualized_volatility, maximum_drawdown, decile_analysis,
    rank_information_coefficient
) 



def trainer(model, train_loader, optimizer, criterion, device, scaler, use_amp, epoch, rank, lambda_return=0.05, lambda_sharpe=0.05):
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    all_predictions = []
    all_labels = []

    if rank == 0:
        print(f"\n[Epoch {epoch+1}] Starting training, total batches: {len(train_loader)}", flush=True)

    for batch_idx, (inputs, labels, target_prices) in enumerate(train_loader):
        start_time = time.time()

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        target_prices = target_prices.to(device, non_blocking=True)
        batch_size = inputs.size(0)
        total_samples += batch_size

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item() * batch_size
        
        # 计算准确率
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(labels).sum().item()
        
        # 收集预测和标签用于计算F1
        all_predictions.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        
        duration = time.time() - start_time

        # 修改：打印 running average 而不是单个 batch
        if rank == 0 and (batch_idx + 1) % 100 == 0:
            running_acc = (total_correct / total_samples) * 100
            running_loss = total_loss / total_samples
            print(f"[Train] Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                  f"Running Loss: {running_loss:.6f}, Running Acc: {running_acc:.2f}%, Time: {duration:.2f}s", flush=True)

    # 计算平均loss和准确率
    metrics = torch.tensor([total_loss, total_samples, total_correct], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    loss_avg = metrics[0].item() / metrics[1].item()
    acc_avg = metrics[2].item() / metrics[1].item() * 100
    
    # 计算详细指标（只在rank 0计算）
    f1_macro = 0.0
    f1_weighted = 0.0
    precision_macro = 0.0
    recall_macro = 0.0
    
    if rank == 0:
        # 计算所有指标
        precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0) * 100
        recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0) * 100
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0) * 100
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100
        f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0) * 100
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        print(f"\n[Epoch {epoch+1}] Training Summary:", flush=True)
        print(f"  Loss: {loss_avg:.6f}", flush=True)
        print(f"  Accuracy: {acc_avg:.2f}%", flush=True)
        print(f"  Precision (Macro): {precision_macro:.2f}%", flush=True)
        print(f"  Recall (Macro): {recall_macro:.2f}%", flush=True)
        print(f"  F1 (Macro): {f1_macro:.2f}%", flush=True)
        print(f"  F1 (Weighted): {f1_weighted:.2f}%", flush=True)
        print(f"  F1 per class - Down: {f1_per_class[0]:.2f}%, Stable: {f1_per_class[1]:.2f}%, Up: {f1_per_class[2]:.2f}%", flush=True)
        print(f"\n  Confusion Matrix:", flush=True)
        print(f"              Pred Down  Pred Stable  Pred Up", flush=True)
        print(f"  True Down      {conf_matrix[0,0]:7d}     {conf_matrix[0,1]:7d}   {conf_matrix[0,2]:7d}", flush=True)
        print(f"  True Stable    {conf_matrix[1,0]:7d}     {conf_matrix[1,1]:7d}   {conf_matrix[1,2]:7d}", flush=True)
        print(f"  True Up        {conf_matrix[2,0]:7d}     {conf_matrix[2,1]:7d}   {conf_matrix[2,2]:7d}", flush=True)

    # 广播指标到所有进程
    metrics_tensor = torch.tensor([loss_avg, acc_avg, precision_macro, recall_macro, f1_macro, f1_weighted], device=device)
    dist.broadcast(metrics_tensor, src=0)
    
    # 解包指标
    loss_avg, acc_avg, precision_macro, recall_macro, f1_macro, f1_weighted = metrics_tensor.tolist()

    return loss_avg, acc_avg, precision_macro, recall_macro, f1_macro, f1_weighted



def evaluator(model, val_loader, criterion, device, use_amp, rank=0, dataset_name="Validation"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    
    # 收集概率、标签和目标价格数据
    all_probs = [] 
    all_labels = []
    all_day128_close = []  # 第128天收盘价 (输入序列最后一天)
    all_day129_close = []  # 第129天收盘价 (目标日)

    if rank == 0:
        print(f"\n[{dataset_name}] Starting evaluation, total batches: {len(val_loader)}", flush=True)

    with torch.no_grad():
        for batch_idx, (inputs, labels, target_prices) in enumerate(val_loader):
            start_time = time.time()

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            target_prices = target_prices.to(device, non_blocking=True)
            batch_size = inputs.size(0)
            total_samples += batch_size

            if use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * batch_size
            
            # 计算标准准确率 (Argmax, 默认阈值逻辑)
            # 获取概率分布
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            
            # 收集评估所需数据
            # inputs shape: [batch, 128, 10] where feature index 3 is close price
            # target_prices shape: [batch, 10] where index 3 is close price
            all_probs.extend(probs.float().cpu().numpy())
            all_labels.extend(labels.cpu().tolist())
            # 第128天的收盘价 (输入序列的最后一天)
            all_day128_close.extend(inputs[:, -1, 3].float().cpu().numpy())
            # 第129天的收盘价 (目标日)
            all_day129_close.extend(target_prices[:, 3].float().cpu().numpy())
            
            duration = time.time() - start_time

            if rank == 0 and (batch_idx + 1) % 50 == 0:
                running_acc = (total_correct / total_samples) * 100
                running_loss = total_loss / total_samples
                print(f"[{dataset_name}] Batch {batch_idx+1}/{len(val_loader)}, "
                      f"Running Loss: {running_loss:.6f}, Running Acc: {running_acc:.2f}%, Time: {duration:.2f}s", flush=True)

    # 同步基础 Loss 和 Accuracy 指标 (所有 GPU)
    metrics = torch.tensor([total_loss, total_samples, total_correct], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    loss_avg = metrics[0].item() / metrics[1].item()
    acc_avg = metrics[2].item() / metrics[1].item() * 100
    
    # 预初始化指标变量
    f1_macro = 0.0
    f1_weighted = 0.0
    precision_macro = 0.0
    recall_macro = 0.0
    
    if rank == 0:
        # 将收集到的数据转为 Numpy 数组方便计算
        np_probs = np.array(all_probs)           # Shape: [N, 3]
        np_labels = np.array(all_labels)         # Shape: [N]
        np_day128_close = np.array(all_day128_close)  # Shape: [N]
        np_day129_close = np.array(all_day129_close)  # Shape: [N]
        
        # 原始的基础预测 (Argmax)
        np_preds = np.argmax(np_probs, axis=1)

        # 计算标准指标
        precision_macro = precision_score(np_labels, np_preds, average='macro', zero_division=0) * 100
        recall_macro = recall_score(np_labels, np_preds, average='macro', zero_division=0) * 100
        f1_macro = f1_score(np_labels, np_preds, average='macro', zero_division=0) * 100
        f1_weighted = f1_score(np_labels, np_preds, average='weighted', zero_division=0) * 100
        f1_per_class = f1_score(np_labels, np_preds, average=None, zero_division=0) * 100
        conf_matrix = confusion_matrix(np_labels, np_preds)
        
        print(f"\n[{dataset_name}] Standard Summary (Argmax):", flush=True)
        print(f"  Loss: {loss_avg:.6f}", flush=True)
        print(f"  Accuracy: {acc_avg:.2f}%", flush=True)
        print(f"  F1 (Macro): {f1_macro:.2f}% | F1 (Weighted): {f1_weighted:.2f}%", flush=True)
        print(f"  F1 per class - Down: {f1_per_class[0]:.2f}%, Stable: {f1_per_class[1]:.2f}%, Up: {f1_per_class[2]:.2f}%", flush=True)
        
        print(f"\n  Confusion Matrix:", flush=True)
        print(f"              Pred Down  Pred Stable  Pred Up", flush=True)
        print(f"  True Down      {conf_matrix[0,0]:7d}     {conf_matrix[0,1]:7d}   {conf_matrix[0,2]:7d}", flush=True)
        print(f"  True Stable    {conf_matrix[1,0]:7d}     {conf_matrix[1,1]:7d}   {conf_matrix[1,2]:7d}", flush=True)
        print(f"  True Up        {conf_matrix[2,0]:7d}     {conf_matrix[2,1]:7d}   {conf_matrix[2,2]:7d}", flush=True)

        # ==========================================================
        # 高置信度交易模拟 (High Confidence Simulation)
        # ==========================================================
        print(f"\n[{dataset_name}] High Confidence Trade Simulation:", flush=True)
        print(f"{'-'*168}")
        print(f"{'Threshold':<11}| {'Class':<7}| {'Signals':<8}| {'Win Rate':<10}| {'Capture':<10}| {'Avg Return':<12}| {'Cum Return':<12}| {'Volatility':<12}| {'Sharpe':<11}| {'Max DD':<11}| {'Profit Factor':<14}")
        print(f"{'-'*168}")

        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        # 计算真实百分比收益: day 129 vs day 128
        # actual_return_pct = (price_129 - price_128) / price_128 * 100
        actual_return_pct = (np_day129_close - np_day128_close) / np_day128_close * 100
        
        # 1 pip spread成本 (约0.01%)
        spread_cost_pct = 0.01
        
        # Hold cost: annual 2% distributed by calendar day
        risk_free_per_trade_pct = (0.02 / 365) * 100
        
        # 无风险利率: 年化 2%
        risk_free_annual = 0.02
        
        for thresh in thresholds:
            # --- 分析 Up (涨) ---
            # 条件: 预测是 Up (索引2) 且 概率 > 阈值
            pred_up_mask = (np_probs[:, 2] > thresh)
            total_up_signals = np.sum(pred_up_mask)
            
            if total_up_signals > 0:
                # 赢: 预测是Up，实际也是Up
                wins = np.sum((np_labels == 2) & pred_up_mask)
                win_rate = (wins / total_up_signals) * 100
                # 捕获率: 实际上有多少个Up被我们抓住了
                total_true_ups = np.sum(np_labels == 2)
                capture_rate = (wins / total_true_ups) * 100 if total_true_ups > 0 else 0
                
                # 对于UP预测（做多/LONG）：
                # return = actual_return - spread
                # - 如果实际上涨(actual_return > 0)，我们获利
                # - 如果实际下跌(actual_return < 0)，我们亏损
                returns_pct = actual_return_pct[pred_up_mask] - spread_cost_pct
                
                cum_return_pct = np.sum(returns_pct)
                avg_return_pct = np.mean(returns_pct)
                volatility_pct = np.std(returns_pct)
                
                # 计算 Sharpe Ratio
                if len(returns_pct) > 1 and volatility_pct > 0:
                    sharpe = sharpe_ratio(
                        returns_pct / 100,
                        risk_free_rate=risk_free_annual,
                        periods_per_year=252
                    )
                else:
                    sharpe = 0.0
                
                # 计算 Maximum Drawdown
                max_dd_pct = maximum_drawdown(returns_pct / 100) * 100
                
                # 计算 Profit Factor (total gains / total losses)
                gains = returns_pct[returns_pct > 0]
                losses = returns_pct[returns_pct < 0]
                profit_factor = np.sum(gains) / abs(np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float('inf')
                
                print(f"{thresh:<11.1f}| {'UP':<7}| {total_up_signals:<8}| {win_rate:8.2f}% | {capture_rate:8.2f}% | {avg_return_pct:+10.4f}% | {cum_return_pct:+10.2f}% | {volatility_pct:10.4f}% | {sharpe:+9.3f} | {max_dd_pct:9.2f}% | {profit_factor:12.2f}")
            else:
                print(f"{thresh:<11.1f}| {'UP':<7}| {0:<8}| {'N/A':<10}| {'N/A':<10}| {'N/A':<12}| {'N/A':<12}| {'N/A':<12}| {'N/A':<11}| {'N/A':<11}| {'N/A':<14}")

            # --- 分析 Down (跌) ---
            # 条件: 预测是 Down (索引0) 且 概率 > 阈值
            # 交易逻辑: SHORT (做空) - 对赌价格下跌
            pred_down_mask = (np_probs[:, 0] > thresh)
            total_down_signals = np.sum(pred_down_mask)
            
            if total_down_signals > 0:
                wins = np.sum((np_labels == 0) & pred_down_mask)
                win_rate = (wins / total_down_signals) * 100
                
                total_true_downs = np.sum(np_labels == 0)
                capture_rate = (wins / total_true_downs) * 100 if total_true_downs > 0 else 0
                
                # 对于DOWN预测（做空/SHORT）：
                # When we predict DOWN and go SHORT:
                # - If price actually goes DOWN (actual_return < 0), we PROFIT: -actual_return > 0
                # - If price actually goes UP (actual_return > 0), we LOSE: -actual_return < 0
                # The returns can be positive or negative, showing realistic P&L
                returns_pct = -actual_return_pct[pred_down_mask] - spread_cost_pct
                
                cum_return_pct = np.sum(returns_pct)
                avg_return_pct = np.mean(returns_pct)
                volatility_pct = np.std(returns_pct)
                
                # 计算 Sharpe Ratio
                if len(returns_pct) > 1 and volatility_pct > 0:
                    sharpe = sharpe_ratio(
                        returns_pct / 100,
                        risk_free_rate=risk_free_annual,
                        periods_per_year=252
                    )
                else:
                    sharpe = 0.0
                
                # 计算 Maximum Drawdown
                max_dd_pct = maximum_drawdown(returns_pct / 100) * 100
                
                # 计算 Profit Factor
                gains = returns_pct[returns_pct > 0]
                losses = returns_pct[returns_pct < 0]
                profit_factor = np.sum(gains) / abs(np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float('inf')
                
                print(f"{'':<11}| {'DOWN':<7}| {total_down_signals:<8}| {win_rate:8.2f}% | {capture_rate:8.2f}% | {avg_return_pct:+10.4f}% | {cum_return_pct:+10.2f}% | {volatility_pct:10.4f}% | {sharpe:+9.3f} | {max_dd_pct:9.2f}% | {profit_factor:12.2f}")
            else:
                print(f"{'':<11}| {'DOWN':<7}| {0:<8}| {'N/A':<10}| {'N/A':<10}| {'N/A':<12}| {'N/A':<12}| {'N/A':<12}| {'N/A':<11}| {'N/A':<11}| {'N/A':<14}")
                
            print(f"{'-'*168}")

        # ==========================================================
        # 资产定价指标: Decile分析 (Asset Pricing Metrics)
        # ==========================================================
        print(f"\n[{dataset_name}] Asset Pricing Metrics - Decile Analysis:", flush=True)
        print(f"{'-'*140}")
        
        up_prob = np_probs[:, 2]
        down_prob = np_probs[:, 0]

        # UP侧: LONG收益（扣spread）
        up_side_returns = (np_day129_close - np_day128_close) / (np_day128_close + 1e-8) * 100 - spread_cost_pct
        # DOWN侧: SHORT收益（扣spread）
        down_side_returns = -(np_day129_close - np_day128_close) / (np_day128_close + 1e-8) * 100 - spread_cost_pct

        def run_decile_table(score, returns, title):
            sorted_indices = np.argsort(score)
            n_samples_decile = len(sorted_indices) // 10

            decile_stats = []
            print(f"\n  {title}", flush=True)
            print(f"{'Decile':<10} | {'Count':<8} | {'Avg Return':<12} | {'Std Dev':<12} | {'Sharpe':<10} | {'Win Rate':<10} | {'Monotonic':<12}")
            print(f"{'-'*140}")

            for decile_idx in range(10):
                start_idx = decile_idx * n_samples_decile
                if decile_idx == 9:
                    end_idx = len(sorted_indices)
                else:
                    end_idx = (decile_idx + 1) * n_samples_decile

                decile_sample_indices = sorted_indices[start_idx:end_idx]
                decile_returns = returns[decile_sample_indices]

                avg_return = np.mean(decile_returns)
                std_dev = np.std(decile_returns)

                if len(decile_returns) > 1 and std_dev > 1e-8:
                    sharpe = sharpe_ratio(
                        decile_returns / 100,
                        risk_free_rate=risk_free_annual,
                        periods_per_year=252
                    )
                else:
                    sharpe = 0.0

                win_rate_decile = (np.sum(decile_returns > 0) / len(decile_returns)) * 100 if len(decile_returns) > 0 else 0

                decile_stats.append({
                    'decile': decile_idx + 1,
                    'count': len(decile_sample_indices),
                    'avg_return': avg_return,
                    'std_dev': std_dev,
                    'sharpe': sharpe,
                    'win_rate': win_rate_decile
                })

                monotonic_check = "↑" if decile_idx > 0 and avg_return > decile_stats[decile_idx-1]['avg_return'] else "→" if decile_idx == 0 else "↓"
                print(f"D{decile_idx+1:<9} | {len(decile_sample_indices):<8} | {avg_return:+11.4f}% | {std_dev:11.4f}% | {sharpe:+9.3f} | {win_rate_decile:9.2f}% | {monotonic_check:<12}")

            print(f"{'-'*140}")

            h_l_spread = decile_stats[9]['avg_return'] - decile_stats[0]['avg_return']
            monotonic_count = sum(1 for i in range(1, 10) if decile_stats[i]['avg_return'] > decile_stats[i-1]['avg_return'])
            best_decile_idx = decile_stats.index(max(decile_stats, key=lambda x: x['sharpe'])) + 1
            worst_decile_idx = decile_stats.index(min(decile_stats, key=lambda x: x['sharpe'])) + 1

            print(f"  {title} Summary:", flush=True)
            print(f"    H-L Spread (Decile 10 - 1): {h_l_spread:+.4f}%", flush=True)
            print(f"    Monotonicity (increasing deciles): {monotonic_count}/9 transitions", flush=True)
            print(f"    Best Decile Sharpe: {max(s['sharpe'] for s in decile_stats):+.3f} (D{best_decile_idx})", flush=True)
            print(f"    Worst Decile Sharpe: {min(s['sharpe'] for s in decile_stats):+.3f} (D{worst_decile_idx})", flush=True)

        run_decile_table(up_prob, up_side_returns, "UP Probability Ranked (LONG Returns)")
        run_decile_table(down_prob, down_side_returns, "DOWN Probability Ranked (SHORT Returns)")
        
        # ==========================================================
        # Additional Comprehensive Metrics
        # ==========================================================
        print(f"\n[{dataset_name}] Comprehensive Metrics:", flush=True)
        print(f"{'-'*80}")
        
        # Cross-entropy loss and McFadden's Pseudo R²
        ce_loss = cross_entropy_loss(np_labels, np_probs[:, 2])  # Using UP class probability
        pseudo_r2 = mcfadden_pseudo_r2(np_labels == 2, np_probs[:, 2])  # Binary: is UP or not
        
        print(f"  Statistical Metrics:", flush=True)
        print(f"    Cross-Entropy Loss (UP class): {ce_loss:.6f}", flush=True)
        print(f"    McFadden's Pseudo R² (UP prediction): {pseudo_r2:.6f}", flush=True)

        # Rank Information Coefficient (trend score vs realized next-step return)
        trend_score = up_prob - down_prob
        rank_ic = rank_information_coefficient(trend_score, actual_return_pct)
        print(f"    Rank IC (trend score vs actual return): {rank_ic:+.6f}", flush=True)
        
        # Long-Short Portfolio Returns (using decile 10 vs decile 1)
        # Construct long-short portfolio
        long_mask = (up_prob >= np.percentile(up_prob, 90))  # Top 10%
        short_mask = (up_prob <= np.percentile(up_prob, 10))  # Bottom 10%
        
        long_returns = up_side_returns[long_mask]
        short_returns = up_side_returns[short_mask]
        
        if len(long_returns) > 0 and len(short_returns) > 0:
            long_short_returns = (long_returns.mean() - short_returns.mean())
            
            # Simulate full long-short portfolio returns over time
            # For simplicity, assume equal weighting across selected assets
            print(f"\n  Long-Short Portfolio (Top 10% vs Bottom 10%):", flush=True)
            print(f"    Long (Top 10%) Avg Return: {long_returns.mean():+.4f}%", flush=True)
            print(f"    Short (Bottom 10%) Avg Return: {short_returns.mean():+.4f}%", flush=True)
            print(f"    Long-Short Spread: {long_short_returns:+.4f}%", flush=True)
            
            # Sharpe ratios
            if len(long_returns) > 1:
                long_sharpe = sharpe_ratio(long_returns / 100, risk_free_rate=risk_free_annual, periods_per_year=252)  # Convert % to decimal
                print(f"    Long Portfolio Sharpe: {long_sharpe:.4f}", flush=True)
            
            if len(short_returns) > 1:
                short_sharpe = sharpe_ratio(-short_returns / 100, risk_free_rate=risk_free_annual, periods_per_year=252)  # Short position
                print(f"    Short Portfolio Sharpe: {short_sharpe:.4f}", flush=True)
        
        # Full portfolio metrics (all predictions)
        all_pred_classes = np.argmax(np_probs, axis=1)
        all_strategy_returns = np.full_like(up_side_returns, np.nan)
        
        # UP predictions: long position
        up_pred_mask = (all_pred_classes == 2)
        all_strategy_returns[up_pred_mask] = up_side_returns[up_pred_mask]
        
        # DOWN predictions: short position
        down_pred_mask = (all_pred_classes == 0)
        all_strategy_returns[down_pred_mask] = down_side_returns[down_pred_mask]
        
        # STABLE predictions: skip (do nothing, no spread deduction)
        stable_pred_mask = (all_pred_classes == 1)
        # Keep as NaN and drop below; no trade, no spread, no hold PnL contribution

        all_strategy_returns = all_strategy_returns[~np.isnan(all_strategy_returns)]
        
        if len(all_strategy_returns) > 1:
            full_sharpe = sharpe_ratio(all_strategy_returns / 100, risk_free_rate=risk_free_annual, periods_per_year=252)
            full_ann_return = annualized_return(all_strategy_returns / 100, periods_per_year=252)
            full_ann_vol = annualized_volatility(all_strategy_returns / 100, periods_per_year=252)
            full_max_dd = maximum_drawdown(all_strategy_returns / 100)
            
            print(f"\n  Full Strategy Portfolio (All Predictions):", flush=True)
            print(f"    Sharpe Ratio (Annualized): {full_sharpe:.4f}", flush=True)
            print(f"    Annualized Return: {full_ann_return*100:.2f}%", flush=True)
            print(f"    Annualized Volatility: {full_ann_vol*100:.2f}%", flush=True)
            print(f"    Maximum Drawdown: {full_max_dd*100:.2f}%", flush=True)
            print(f"    Total Cumulative Return: {np.sum(all_strategy_returns):.2f}%", flush=True)
            print(f"    Mean Return per Trade: {np.mean(all_strategy_returns):.4f}%", flush=True)
            print(f"    Win Rate (Positive Returns): {(np.sum(all_strategy_returns > 0) / len(all_strategy_returns)) * 100:.2f}%", flush=True)
        
        print(f"{'-'*80}")

    # 广播基础指标到所有进程 (保持流程完整性)
    metrics_tensor = torch.tensor([loss_avg, acc_avg, precision_macro, recall_macro, f1_macro, f1_weighted], device=device)
    dist.broadcast(metrics_tensor, src=0)
    
    # 解包
    loss_avg, acc_avg, precision_macro, recall_macro, f1_macro, f1_weighted = metrics_tensor.tolist()

    return loss_avg, acc_avg, precision_macro, recall_macro, f1_macro, f1_weighted