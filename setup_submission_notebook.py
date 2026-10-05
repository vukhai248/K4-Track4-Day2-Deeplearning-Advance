"""Update the existing notebook without changing original evaluator or source lab docs."""
from pathlib import Path
import ast
import json

root = Path(__file__).resolve().parent
path = root / 'Colab_DeepWeeds.ipynb'
nb = json.loads(path.read_text(encoding='utf-8'))
nb['cells'] = [cell for cell in nb['cells'] if not cell.get('metadata', {}).get('day2_git_preflight')]
def source(index): return ''.join(nb['cells'][index]['source'])
def put(index, text): nb['cells'][index]['source'] = text.splitlines(keepends=True)

# Conservative batch; common recipe across backbones.
put(10, source(10).replace('BATCH_SIZE = 128 if is_large_gpu else 32', 'BATCH_SIZE = 32'))
download_cell = source(6).replace('import os, zipfile, hashlib, urllib.request',
                                  'import os, zipfile, hashlib, urllib.request, subprocess')
download_cell = download_cell.replace(
    "os.system(f'wget -q --show-progress -O \"{IMAGES_ZIP}\" \"{IMAGES_URL}\"')",
    "subprocess.run(['wget', '-q', '-O', str(IMAGES_ZIP), IMAGES_URL], check=True)")
retry_on_bad_checksum = """print('⚠️ Checksum sai; tải lại images.zip một lần (vẫn ẩn thanh tiến độ).')
        IMAGES_ZIP.unlink(missing_ok=True)
        subprocess.run(['wget', '-q', '-O', str(IMAGES_ZIP), IMAGES_URL], check=True)
        if not check_md5(IMAGES_ZIP, IMAGES_MD5):
            raise RuntimeError('images.zip checksum van sai sau lan tai lai; dung file nay.')
        print('✅ Checksum khớp sau khi tải lại.')"""
download_cell = download_cell.replace(
    "print('⚠️ Cảnh báo: Checksum MD5 không khớp hoặc file tải bị gián đoạn.')", retry_on_bad_checksum)
download_cell = download_cell.replace(
    "raise RuntimeError('images.zip checksum khong khop; tai lai file truoc khi giai nen.')",
    retry_on_bad_checksum)
put(6, download_cell)
sanity_source = source(8)
if '# SANITY AUGMENTATION PREVIEW' not in sanity_source:
    sanity_source += '''

# Save quantitative preflight evidence; package() includes eval_out JSON files.
Path('eval_out').mkdir(exist_ok=True)
Path('eval_out/sanity_checks.json').write_text(json.dumps({
    'initial_ce': float(init_loss), 'uniform_reference_ln9': float(expected_loss),
    'initial_loss_close_to_ln9': bool(abs(init_loss - expected_loss) < 0.6),
    'same_batch_ce_after_30_updates': float(loss.item()),
    'same_batch_overfit_threshold_pass': bool(loss.item() < 0.2),
    'seed': 0, 'fold': 0}, indent=2), encoding='utf-8')

# SANITY AUGMENTATION PREVIEW — save evidence for the lab report/package.
from PIL import Image
fig, axs = plt.subplots(2, 4, figsize=(13, 6))
preview_tf = ds.build_transforms(train=True, img_size=224, aug='color')
for col, (_, sample) in enumerate(train_df.head(4).iterrows()):
    image = Image.open(IMAGES_DIR / sample.Filename).convert('RGB')
    axs[0, col].imshow(image); axs[0, col].set_title('Original')
    tensor = preview_tf(image).permute(1, 2, 0).numpy()
    tensor = np.clip(tensor * np.array(ds.IMAGENET_STD) + np.array(ds.IMAGENET_MEAN), 0, 1)
    axs[1, col].imshow(tensor); axs[1, col].set_title('Color aug')
    for row in range(2): axs[row, col].axis('off')
fig.suptitle('Training augmentation preview (four Fold 0 train images)')
fig.tight_layout()
Path('curves/augmentation_preview.png').parent.mkdir(parents=True, exist_ok=True)
fig.savefig('curves/augmentation_preview.png', dpi=150)
plt.show(); plt.close(fig)
print('Saved augmentation evidence: curves/augmentation_preview.png')
'''
put(8, sanity_source)
# Three ablation axes with baseline + at least two alternatives each.
s = source(12).replace("TARGET_BB = best_bb if 'best_bb' in locals() else 'convnext_tiny'", 'TARGET_BB = best_bb')
if "('T08'," not in s:
    s = s.replace("    ('T05',", "    ('T08', 'Khởi tạo', 'From scratch', {'init': 'scratch'}),\n    ('T05',")
s = s.replace("best_recipe_exp = df_ablation.sort_values(by='val_macro_f1', ascending=False).iloc[0]['exp_id']", """baseline_exp = df_backbones[df_backbones.backbone == TARGET_BB].iloc[0].exp_id
recipe_candidates = pd.concat([df_ablation[['exp_id', 'val_macro_f1']],
    pd.DataFrame([{'exp_id': baseline_exp, 'val_macro_f1': baseline_f1}])], ignore_index=True)
best_recipe_exp = recipe_candidates.sort_values('val_macro_f1', ascending=False).iloc[0].exp_id
f01_overrides = next((overrides.copy() for exp_id, _, _, overrides in ablation_configs
                      if exp_id == best_recipe_exp), {})""")
s = s.replace("baseline_f1 = baseline_row['val_macro_f1'] if baseline_row is not None else 0.92", "baseline_f1 = float(baseline_row['val_macro_f1'])")
put(12, s)
put(14, '''import inference as inf
import eval as ev
import json
from contextlib import nullcontext

best_model_path = Path('runs') / best_recipe_exp / 'seed0' / 'best_model.pth'
eval_model = md.build_model(TARGET_BB, pretrained=False, num_classes=ds.NUM_CLASSES)
eval_model.load_state_dict(torch.load(best_model_path, map_location='cpu', weights_only=True))
eval_model.to(device).eval()
val_loader_224 = ds.make_loader(val_df, IMAGES_DIR, ds.build_transforms(train=False, img_size=224),
                               batch_size=32, train=False, num_workers=2)
def logits_to_probs(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
def measure(views, batch=1):
    dummy = torch.randn(batch, 3, 224, 224, device=device)
    def forward():
        with torch.autocast(device_type='cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext():
            for view in views: eval_model(view(dummy))
    with torch.inference_mode():
        report = bm.bench(forward, warmup=10, iters=50,
                          sync=torch.cuda.synchronize if device.type == 'cuda' else None)
    report.update(gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
                  dtype='amp' if device.type == 'cuda' else 'fp32', batch=batch,
                  images_per_s=batch * 1000 / report['p50'])
    return report
identity = lambda x: x
vertical = lambda x: torch.flip(x, dims=[-2])
fns, y_val_true, val_logits = inf.predict_logits(eval_model, val_loader_224, device)
_, _, horizontal_logits = inf.predict_logits(eval_model, val_loader_224, device, view=inf.view_hflip)
_, _, vertical_logits = inf.predict_logits(eval_model, val_loader_224, device, view=vertical)
def softmax_np(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
logit_tta = softmax_np((val_logits + horizontal_logits) / 2.0)
T_opt = inf.fit_temperature(val_logits, y_val_true)
lat_i00 = measure([identity])
lat_i01 = measure([identity, inf.view_hflip])
lat_i02 = measure([identity, vertical])
lat_b32 = measure([identity], batch=32)
methods = [
    ('I00', '1-view 224', logits_to_probs(val_logits), lat_i00, 1),
    ('I01', 'TTA horizontal: probability average', inf.aggregate_views([val_logits, horizontal_logits]), lat_i01, 2),
    ('I02', 'TTA vertical: probability average', inf.aggregate_views([val_logits, vertical_logits]), lat_i02, 2),
    ('I03', 'TTA horizontal: logit average', logit_tta, lat_i01, 2),
    ('I04', f'Temperature scaling T={T_opt:.4f}', inf.apply_temperature(val_logits, T_opt), lat_i00, 1)]
rows = []
for exp_id, method, probabilities, latency, k in methods:
    metrics = ev.compute_metrics(y_val_true, probabilities.argmax(axis=1), probabilities)
    rows.append(dict(exp_id=exp_id, method=method, model_ckpt=str(best_model_path), k_views=k,
        val_macro_f1=metrics['macro_f1'], val_top1=metrics['top1'], val_ece=metrics['ece'],
        latency_p50_ms=latency['p50'], latency_p95_ms=latency['p95'], latency_p99_ms=latency['p99'],
        throughput_img_s=latency['images_per_s'], relative_cost=latency['p50']/lat_i00['p50'],
        gpu=latency['gpu'], dtype=latency['dtype'], batch=1))
df_inference = pd.DataFrame(rows)
display(df_inference)
ece_i00 = float(df_inference.loc[df_inference.exp_id == 'I00', 'val_ece'].iloc[0])
ece_calibrated = float(df_inference.loc[df_inference.exp_id == 'I04', 'val_ece'].iloc[0])
print('TEST will be evaluated once after fitting per-seed temperature on VAL only.')
del eval_model
torch.cuda.empty_cache()
''')
s = source(16)
s = s.replace("f01_overrides = {'mix': 'cutmix', 'mix_alpha': 1.0, 'loss': 'ls', 'label_smoothing': 0.1, 'ema_decay': 0.999}", "print('Final recipe selected on VAL:', best_recipe_exp, f01_overrides)")
# Run original scoring tool strictly; preserve output for submission evidence.
s = s.split("print('\\n=== CHẠY EVAL.PY SCORE ===')")[0].split('def run_eval(')[0] + '''
def run_eval(args, filename):
    result = subprocess.run([sys.executable, 'eval.py', *args], capture_output=True, text=True)
    Path('eval_out').mkdir(exist_ok=True)
    Path('eval_out', filename).write_text(result.stdout + result.stderr, encoding='utf-8')
    print(result.stdout)
    if result.returncode: raise RuntimeError(result.stderr)
common = ['--test-csv', 'data/labels/test_subset0.csv', '--labels', 'data/labels/labels.csv']
# Temperature is fitted independently on each final seed's VAL probabilities only.
# Existing TEST probabilities are calibrated without another model pass.
import shutil
calibration_rows = []
prob_cols = [f'p{i}' for i in range(ds.NUM_CLASSES)]
for seed in SEEDS:
    val_path = Path(f'predictions/F01_seed{seed}_val.csv')
    test_path = Path(f'predictions/F01_seed{seed}_test.csv')
    uncal_path = Path(f'predictions/F01_seed{seed}_uncal_test.csv')
    val_frame, test_frame = pd.read_csv(val_path), pd.read_csv(test_path)
    val_logits_seed = np.log(np.clip(val_frame[prob_cols].to_numpy(), 1e-12, 1.0))
    temperature = inf.fit_temperature(val_logits_seed, val_frame.y_true.to_numpy())
    shutil.copy2(test_path, uncal_path)
    calibrated = inf.apply_temperature(np.log(np.clip(test_frame[prob_cols].to_numpy(), 1e-12, 1.0)), temperature)
    test_frame[prob_cols] = calibrated
    test_frame['y_pred'] = calibrated.argmax(axis=1)
    test_frame.to_csv(test_path, index=False)
    calibration_rows.append({'seed': seed, 'temperature_fit_on': 'VAL only', 'temperature': temperature,
        'test_ece_before': ev.ece_score(pd.read_csv(uncal_path)[prob_cols].to_numpy(),
                                        pd.read_csv(uncal_path).y_true.to_numpy()),
        'test_ece_after': ev.ece_score(calibrated, test_frame.y_true.to_numpy())})
Path('eval_out/temperature_scaling.json').write_text(json.dumps(calibration_rows, indent=2), encoding='utf-8')
print('Per-seed temperatures fitted on VAL; TEST predictions calibrated without another forward pass.')
for tag in ['F01', 'T00']:
    run_eval(['score', '--pred', f'predictions/{tag}_seed*_test.csv', *common,
              '--tag', tag, '--out', 'eval_out'], tag + '_score.txt')
run_eval(['grade', '--final', 'predictions/F01_seed*_test.csv',
          '--baseline', 'predictions/T00_seed*_test.csv', *common,
          '--uncal', 'predictions/F01_seed*_uncal_test.csv',
          '--final-val', 'predictions/F01_seed*_val.csv',
          '--val-csv', 'data/labels/val_subset0.csv', '--out', 'eval_out',
          '--latency-p95-ms', str(lat_i00['p95'])], 'grade.txt')
'''
put(16, s)
put(18, "from lab_outputs import export\nsubmission_tables = export(globals())\n")
put(20, '''from submission import package, push_submission
submission_dir = package()
try:
    submission_pr_url = push_submission(submission_dir)
except Exception as error:
    print('Push/PR failed:', type(error).__name__, str(error))
    print('The ZIP is already saved. Fix credentials/permissions and rerun this cell only.')
''')
put(22, '''from pathlib import Path
from submission import STUDENT
zip_path = (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()) / (STUDENT + '_submission.zip')
print('Submission ZIP:', zip_path)
try:
    from google.colab import files
    files.download(str(zip_path))
except ImportError:
    print('Kaggle: Save Version to preserve Output, then download this ZIP from Output.')
''')
# Match visible descriptions to final implementation.
put(0, '''# DeepWeeds — Track 4 Day 2 — Vu Gia Khai (2A202602786)
GPU + Internet ON. Run all cells in order. Add-ons > Secrets: GITHUB_TOKEN (optional for ZIP, required for Git).
After real experiments: results.xlsx, report.md, plots, predictions, code and logs are packaged under
submissions/2A202602786_Vu_Gia_Khai/. A draft PR is created if GitHub credentials permit.
Read report.md and review the evidence before submitting the PR to your instructor.
''')
put(13, '## Cell 7 — Four inference methods beyond I00 on VAL\nI01 horizontal TTA with probability averaging; I02 vertical TTA with probability averaging; I03 horizontal TTA with logit averaging; I04 temperature scaling. 224 px; latency uses 10 warmups and 50 synchronized repeats.\n')
put(19, '## Cell 10 — Package submission, then Git push and draft PR\nRequires completed real results. No token in Git URL; uses Kaggle/Colab Secrets. Missing token keeps the ZIP available.\n')
bootstrap = source(2).split('# Use the reviewed student implementation')[0].rstrip() + '\n'
bootstrap = bootstrap.replace('import os, sys\n', 'import os, sys, subprocess, json\n')
bootstrap = bootstrap.replace("os.system(f'git clone {REPO_URL} \"{TARGET_DIR}\"')", "subprocess.run(['git', 'clone', REPO_URL, str(TARGET_DIR)], check=True)")
runtime_files = {p.relative_to(root).as_posix(): p.read_text(encoding='utf-8') for p in (root / 'code').glob('*.py')}
bootstrap += '\n# Use the reviewed student implementation, rather than stale remote code.\n'
bootstrap += 'RUNTIME_FILES = ' + repr(runtime_files) + '\n'
bootstrap += "for relative, text in RUNTIME_FILES.items():\n    dest = Path(relative)\n    dest.parent.mkdir(parents=True, exist_ok=True)\n    dest.write_text(text, encoding='utf-8')\n"
put(2, bootstrap)
nb['cells'][3:3] = [
    {'cell_type': 'markdown', 'metadata': {'day2_git_preflight': True}, 'source':
     ['## Git preflight — run again after adding GITHUB_TOKEN\n',
      'Tests Git push with --dry-run. Does not train or create a branch/PR. Missing token does not block ZIP export.\n']},
    {'cell_type': 'code', 'metadata': {'day2_git_preflight': True}, 'execution_count': None,
     'outputs': [], 'source': ['from git_preflight import check\n', 'git_ready = check()\n']}]
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
snapshot = json.dumps(nb, ensure_ascii=False)
put(2, bootstrap + "\nPath('Colab_DeepWeeds.ipynb').write_text(" + repr(snapshot) + ", encoding='utf-8')\n")
for cell in nb['cells']:
    if cell['cell_type'] == 'code': ast.parse(''.join(cell['source']))
path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"Updated notebook; all {sum(c['cell_type'] == 'code' for c in nb['cells'])} code cells pass syntax checks.")
