"""Build submission tables and an evidence-based report from completed runs."""
from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from PIL import Image


def export(ns):
    import torch
    import timm
    import eval as ev
    ds = ns['ds']
    curves = Path('curves')
    curves.mkdir(exist_ok=True)
    final_rows, per_class = [], []
    for exp in ['F01', 'T00']:
        for seed in [0, 1, 2]:
            pred = pd.read_csv(f'predictions/{exp}_seed{seed}_test.csv')
            val = pd.read_csv(f'predictions/{exp}_seed{seed}_val.csv')
            probs = [f'p{i}' for i in range(ds.NUM_CLASSES)]
            metrics = ev.compute_metrics(pred.y_true.values, pred.y_pred.values, pred[probs].values)
            val_metrics = ev.compute_metrics(val.y_true.values, val.y_pred.values, val[probs].values)
            cfg = json.loads(Path(f'runs/{exp}/seed{seed}/config.json').read_text())
            final_rows.append(dict(exp_id=exp, seed=seed, config=json.dumps(cfg, sort_keys=True),
                                   inference=('I04: temperature scaling fitted on VAL' if exp == 'F01' else 'I00: single view'),
                                   val_macro_f1=val_metrics['macro_f1'],
                                   test_macro_f1=metrics['macro_f1'], test_top1=metrics['top1'], test_ece=metrics['ece']))
            precision, recall, f1, support = precision_recall_fscore_support(
                pred.y_true, pred.y_pred, labels=range(ds.NUM_CLASSES), zero_division=0)
            per_class.extend(dict(exp_id=exp, seed=seed, class_id=i, class_name=name,
                                  n_test=int(support[i]), precision=precision[i], recall=recall[i], f1=f1[i])
                             for i, name in enumerate(ds.CLASS_NAMES))
    final = pd.DataFrame(final_rows)
    summaries = final.groupby('exp_id')[['test_macro_f1', 'test_top1', 'test_ece']].agg(['mean', 'std'])
    final_display = final.copy()
    for exp in ['F01', 'T00']:
        row = {'exp_id': exp + '_mean_std', 'seed': '0,1,2', 'config': exp,
               'inference': 'I04 (F01) / I00 (T00)'}
        for col in ['test_macro_f1', 'test_top1', 'test_ece']:
            row[col] = f"{summaries.loc[exp, (col, 'mean')]:.6f} ± {summaries.loc[exp, (col, 'std')]:.6f}"
        final_display = pd.concat([final_display, pd.DataFrame([row])], ignore_index=True)
    backbones = ns['df_backbones'].copy()
    backbones['train_time_per_epoch_min'] = backbones.train_time_min / backbones.epochs
    training = ns['df_ablation'].copy()
    inference = ns['df_inference'].copy()
    latency = pd.DataFrame([dict(config=ns['TARGET_BB'], **ns['lat_i00']),
                            dict(config=ns['TARGET_BB'], **ns['lat_b32'])])
    combined = pd.concat([backbones.assign(stage='backbone'), training.assign(stage='training'),
                          inference.assign(stage='inference')], ignore_index=True)
    summary = combined.sort_values('val_macro_f1', ascending=False).head(10)
    tables = {'Backbones': backbones, 'Training': training, 'Inference': inference,
              'Final': final_display, 'PerClass': pd.DataFrame(per_class),
              'Latency': latency, 'Summary': summary}
    with pd.ExcelWriter('results.xlsx', engine='openpyxl') as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(55, max(14, len(str(column[0].value)) + 3))

    # Rebuild distinct seed curves from real logs, avoiding overwritten images.
    for history_path in Path('runs').glob('*/*/history.csv'):
        history = pd.read_csv(history_path)
        exp, seed = history_path.parts[-3:-1]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history.epoch, history.train_loss, label='train loss')
        ax[0].plot(history.epoch, history.val_loss, label='val loss')
        ax[1].plot(history.epoch, history.val_macro_f1, label='val macro-F1')
        ax[1].plot(history.epoch, history.val_acc, label='val accuracy')
        for axis in ax:
            axis.set_xlabel('Epoch')
            axis.legend()
        fig.suptitle(exp + ' / ' + seed)
        fig.tight_layout()
        fig.savefig(curves / f'{exp}_{seed}.png', dpi=150)
        plt.close(fig)
    pred = pd.read_csv('predictions/F01_seed0_test.csv')
    cm = confusion_matrix(pred.y_true, pred.y_pred, labels=range(ds.NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(9), ds.CLASS_NAMES, rotation=60, ha='right')
    ax.set_yticks(range(9), ds.CLASS_NAMES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(9):
        for j in range(9):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7)
    fig.tight_layout(); fig.savefig(curves / 'confusion_matrix_test.png', dpi=150); plt.close(fig)
    mistakes = pred[pred.y_true != pred.y_pred].head(12)
    if not mistakes.empty:
        fig, axs = plt.subplots(3, 4, figsize=(13, 10))
        for axis in axs.flat: axis.axis('off')
        for axis, (_, row) in zip(axs.flat, mistakes.iterrows()):
            with Image.open(Path(ns['IMAGES_DIR']) / row.Filename) as image:
                axis.imshow(image.convert('RGB'))
            axis.set_title(f"{row.Filename}\n{ds.CLASS_NAMES[int(row.y_true)]} → {ds.CLASS_NAMES[int(row.y_pred)]}", fontsize=7)
        fig.tight_layout(); fig.savefig(curves / 'misclassified_test_examples.png', dpi=150); plt.close(fig)
    counts = pd.DataFrame({name: frame.Label.value_counts().reindex(range(9), fill_value=0)
                           for name, frame in [('train', ns['train_df']), ('val', ns['val_df']), ('test', ns['test_df'])]})
    counts.index = ds.CLASS_NAMES
    counts.plot.bar(figsize=(11, 5)); plt.ylabel('Images'); plt.tight_layout()
    plt.savefig(curves / 'class_distribution.png', dpi=150); plt.close()
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in inference.iterrows():
        ax.scatter(row.latency_p50_ms, row.val_macro_f1)
        ax.annotate(row.exp_id, (row.latency_p50_ms, row.val_macro_f1))
    ax.set_xlabel('p50 latency (ms)'); ax.set_ylabel('VAL Macro-F1')
    fig.tight_layout(); fig.savefig(curves / 'accuracy_latency.png', dpi=150); plt.close(fig)

    # Markdown tables without requiring the optional tabulate package.
    def table(frame):
        def clean(value): return str(value).replace('|', '/').replace('\n', ' ')
        return '\n'.join(['| ' + ' | '.join(map(clean, frame.columns)) + ' |',
                           '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                          ['| ' + ' | '.join(map(clean, values)) + ' |' for values in frame.itertuples(index=False, name=None)])
    f_mean = summaries.loc['F01', ('test_macro_f1', 'mean')]
    b_mean = summaries.loc['T00', ('test_macro_f1', 'mean')]
    noise = max(summaries.loc[exp, ('test_macro_f1', 'std')] for exp in ['F01', 'T00'])
    delta = f_mean - b_mean
    off_diag = cm.copy(); np.fill_diagonal(off_diag, 0)
    a, b = np.unravel_index(off_diag.argmax(), off_diag.shape)
    notes = ('Mức tăng vượt std lớn hơn của hai nhóm seed.' if delta > noise else
             'Chênh lệch không vượt std lớn hơn của hai nhóm seed; chưa đủ bằng chứng kết luận cải thiện ổn định.')
    best_axis = training[training.axis != 'Kết hợp'].sort_values('delta_vs_t00', ascending=False).iloc[0]
    config = Path('runs/F01/seed0/config.json').read_text()
    calibration = json.loads(Path('eval_out/temperature_scaling.json').read_text(encoding='utf-8'))
    ece_before_mean = float(np.mean([row['test_ece_before'] for row in calibration]))
    ece_after_mean = float(np.mean([row['test_ece_after'] for row in calibration]))
    report = f'''# DeepWeeds — Track 4 Day 2
Vu Gia Khai — MSSV 2A202602786

## 1. Tóm tắt
So sánh 5 backbone, các công thức huấn luyện trên VAL và 4 phương pháp suy luận.
F01 dùng {ns['TARGET_BB']}, công thức {ns['best_recipe_exp']}, suy luận I04 (temperature scaling; T khớp riêng trên VAL từng seed). T00 dùng I00.
TEST Macro-F1 trung bình {f_mean:.6f}; baseline {b_mean:.6f}; Δ={delta:.6f}.
{notes} Các con số được tính từ CSV dự đoán thật; xem bảng Final để có mean ± std đầy đủ.

## 2. Dữ liệu và thiết lập
DeepWeeds 17.509 ảnh, 9 lớp, official fold 0, seeds 0/1/2 cho chung kết và baseline.
Không dùng TEST để chọn backbone, recipe hoặc inference. Chỉ VAL dùng để lựa chọn.
GPU: {ns['lat_i00']['gpu']}. Python {platform.python_version()}, PyTorch {torch.__version__}, timm {timm.__version__}.
Các phiên bản đầy đủ và hyperparameters được lưu trong requirements-lock.txt và logs/.
![Phân bố lớp](curves/class_distribution.png)
![Ảnh gốc và augmentation](curves/augmentation_preview.png)
Các kiểm tra loss ban đầu và overfit một batch được lưu ở `evaluation/sanity_checks.json`.

## 3. So sánh backbone
{table(backbones)}
Chọn backbone có VAL Macro-F1 cao nhất theo số đo trên cùng split, seed, epochs và công thức nền.
GMAC từ bộ đếm hooks là ước lượng, đặc biệt có thể thiếu chi phí attention của transformer.

## 4. Công thức huấn luyện
{table(training)}
Thay đổi đơn yếu tố tốt nhất trong các dòng đã chạy: {best_axis.exp_id}, Δ VAL={best_axis.delta_vs_t00:.6f}.
Các ablation chỉ có một seed nên chưa chứng minh độ ổn định; dòng kết hợp không dùng để quy công cho một yếu tố riêng lẻ.

## 5. Suy luận và hiệu chuẩn
{table(inference)}
![Đánh đổi](curves/accuracy_latency.png)
Chung kết dùng I04: nhiệt độ được khớp riêng trên VAL cho từng seed rồi áp dụng vào xác suất TEST đã lưu,
không chạy thêm forward và không dùng nhãn TEST để chọn T. predictions/*_uncal_test.csv giữ xác suất gốc để đối chiếu ECE.
ECE TEST trung bình trước/sau temperature scaling lần lượt là {ece_before_mean:.6f} / {ece_after_mean:.6f}; đối chiếu tự chấm ở evaluation/grade_I.json.
Các phương pháp nhiều view phù hợp hơn khi không bị giới hạn thời gian.

## 6. Chung kết, baseline và phân tích lỗi
{table(final_display.drop(columns=['config']))}
![Ma trận nhầm lẫn](curves/confusion_matrix_test.png)
![Ảnh đoán sai](curves/misclassified_test_examples.png)
Cặp nhầm có hướng nhiều nhất ở seed 0: {ds.CLASS_NAMES[a]} → {ds.CLASS_NAMES[b]} ({off_diag[a, b]} ảnh).
Chinee Apple → Snake Weed: {cm[0, 7]}; chiều ngược lại: {cm[7, 0]}.
Các ảnh trên là ví dụ theo thứ tự CSV, không phải mẫu đại diện ngẫu nhiên. Hình dạng lá, nền thực vật và ánh sáng
là các giả thuyết về nguyên nhân nhầm lẫn, chưa có kiểm chứng nhân quả. F1 từng lớp và baseline nằm trong sheet PerClass.

## 7. Kết luận và khuyến nghị
Δ TEST F01 so với T00 là {delta:.6f}; std lớn hơn giữa hai nhóm là {noise:.6f}. {notes}
Không thể tách hoàn toàn đóng góp backbone và recipe từ phép so sánh TEST; đối chiếu các ablation VAL riêng ở trên.
I00 p95={ns['lat_i00']['p95']} ms chỉ đo model forward trên GPU được ghi, chưa gồm tải ảnh và tiền xử lý.
Với ngân sách robot 30–100 ms/khung, cần benchmark toàn pipeline trên phần cứng triển khai trước khi kết luận đạt yêu cầu.

## 8. Hạn chế
Một fold; ablation một seed; chung kết ba seed; chia ngẫu nhiên có thể lạc quan so với chia theo địa điểm.
Không suy rộng sang miền, mùa hoặc thiết bị khác. Chưa thực hiện thử nghiệm triển khai hoặc kiểm định thống kê chính thức.
Số epoch 12 là giới hạn ngân sách; chỉ sử dụng một GPU của phiên T4 x2.
Đọc lại nhận xét về ảnh lỗi và đường cong trước khi đánh dấu PR sẵn sàng nộp.

## 9. Phụ lục cấu hình chung kết
```json
{config}
```
Các config/history của từng exp_id nằm trong logs/. Notebook và cách chạy nằm trong README.md.
Dataset: https://zenodo.org/records/7939060 ; không commit dataset hoặc checkpoint.
'''
    Path('report.md').write_text(report, encoding='utf-8')
    print('Đã xuất results.xlsx và report.md từ dữ liệu thật. Đọc lại báo cáo trước khi nộp chính thức.')
    return tables
