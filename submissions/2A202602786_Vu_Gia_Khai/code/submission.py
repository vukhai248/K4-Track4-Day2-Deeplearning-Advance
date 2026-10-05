"""Package real Day 2 outputs and push only the student's submission directory."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
from datetime import datetime, timezone

STUDENT = '2A202602786_Vu_Gia_Khai'
NOTEBOOK_URL = 'https://www.kaggle.com/code/b22dckh065vgiakhi/notebooke997c35844'
REPO = 'vukhai248/K4-Track4-Day2-Deeplearning-Advance'


def get_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret('GITHUB_TOKEN')
    except Exception:
        try:
            from google.colab import userdata
            return userdata.get('GITHUB_TOKEN')
        except Exception:
            return None


def github_api(path, token, payload=None):
    headers = {'Authorization': 'Bearer ' + token,
               'Accept': 'application/vnd.github+json', 'User-Agent': 'Day2-Submission'}
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request('https://api.github.com/' + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def package(root='.'):
    import pandas as pd
    root = Path(root).resolve()
    dest = root / 'submissions' / STUDENT
    dest.mkdir(parents=True, exist_ok=True)
    required = ['results.xlsx', 'report.md', 'eval.py'] + [
        f'predictions/{exp}_seed{seed}_{split}.csv'
        for exp in ['F01', 'T00'] for seed in range(3) for split in ['test', 'val']]
    required += [f'predictions/F01_seed{seed}_uncal_test.csv' for seed in range(3)]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError('Chua du ket qua that: ' + ', '.join(missing))
    sheets = pd.read_excel(root / 'results.xlsx', sheet_name=None)
    expected = {'Backbones', 'Training', 'Inference', 'Final', 'PerClass', 'Latency', 'Summary'}
    if not expected.issubset(sheets) or any(sheets[name].empty for name in expected):
        raise RuntimeError('results.xlsx phai co du 7 sheets voi du lieu that.')
    report = (root / 'report.md').read_text(encoding='utf-8')
    if 'TODO' in report or len(report) < 1000:
        raise RuntimeError('report.md chua hoan thien; doc va bo sung phan TODO truoc khi nop.')
    for name in ['results.xlsx', 'report.md']:
        shutil.copy2(root / name, dest / name)
    # The official evaluator is submitted unchanged so the work is reproducible.
    (dest / 'code').mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / 'eval.py', dest / 'code' / 'eval.py')
    for folder, suffixes in [('code', {'.py', '.ipynb'}), ('curves', {'.png'}),
                             ('predictions', {'.csv'})]:
        for src in (root / folder).rglob('*'):
            if src.is_file() and src.suffix in suffixes and '__pycache__' not in src.parts:
                target = dest / folder / src.relative_to(root / folder)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
    # Small reproducibility evidence; no datasets or model checkpoints.
    for src in (root / 'runs').glob('*/*/*'):
        if src.name in {'config.json', 'history.csv'}:
            target = dest / 'logs' / src.relative_to(root / 'runs')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    for src in (root / 'eval_out').glob('*'):
        if src.is_file() and src.suffix in {'.json', '.csv', '.txt'}:
            (dest / 'evaluation').mkdir(exist_ok=True)
            shutil.copy2(src, dest / 'evaluation' / src.name)
    for src in root.glob('Colab_DeepWeeds*.ipynb'):
        notebook = json.loads(src.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
        (dest / 'code' / src.name).write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    versions = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True,
                              text=True, check=True).stdout
    (dest / 'requirements-lock.txt').write_text(versions, encoding='utf-8')
    readme = f'''# Lab Day 2 — Vu Gia Khai — 2A202602786

Notebook: {NOTEBOOK_URL}

Run Colab_DeepWeeds.ipynb from top to bottom on GPU with Internet enabled.
It downloads DeepWeeds, compares five backbones, ablates training recipes,
compares I01-I04 against I00, then evaluates F01 and T00 with seeds 0, 1, 2.
Select all configurations on VAL before opening TEST. Do not rerun TEST to tune.
Full versions are in requirements-lock.txt. Logs/configurations are in logs/.
The original eval.py at repository root is unchanged.

Submission: results.xlsx, report.md, curves/, predictions/, code/ (including unchanged eval.py).
Checkpoint files remain in Kaggle Output under runs/; they are not committed.
To rerun just packaging and Git after reviewing report.md: run the last two cells.
Use Kaggle Add-ons > Secrets > GITHUB_TOKEN for GitHub authentication.
Never put a token into notebook source. See report.md for setup and limitations.
'''
    (dest / 'README.md').write_text(readme, encoding='utf-8')
    output_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else root
    archive = shutil.make_archive(str(output_dir / (STUDENT + '_submission')), 'zip',
                                  root_dir=root, base_dir=str(dest.relative_to(root)))
    print('Submission folder:', dest)
    print('ZIP:', archive)
    return dest


def push_submission(dest, token=None):
    token = token or get_token()
    if not token:
        print('Chua push: them GITHUB_TOKEN trong Add-ons > Secrets, roi chay lai cell nay.')
        return None
    info = github_api('repos/' + REPO, token)
    if not info.get('permissions', {}).get('push'):
        raise RuntimeError('Token/account khong co quyen ghi repo. Can quyen ghi hoac luong fork + PR.')
    user = github_api('user', token)
    branch = 'results/kaggle-' + STUDENT + '-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    url = 'https://github.com/' + REPO + '.git'
    with tempfile.TemporaryDirectory() as tmp:
        askpass = Path(tmp) / 'askpass.py'
        askpass.write_text('#!/usr/bin/env python3\nimport os,sys\nprint("x-access-token" if "username" in sys.argv[1].lower() else os.environ["DAY2_GIT_TOKEN"])\n')
        askpass.chmod(0o700)
        env = dict(os.environ, DAY2_GIT_TOKEN=token, GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT='0')
        checkout = Path(tmp) / 'repo'
        def git(*args):
            result = subprocess.run(['git', *args], env=env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError((result.stderr or result.stdout).replace(token, '[REDACTED]'))
            return result.stdout
        git('clone', '--depth', '1', '--branch', info['default_branch'], url, str(checkout))
        target = checkout / 'submissions' / STUDENT
        shutil.copytree(dest, target, dirs_exist_ok=True)
        git('-C', str(checkout), 'checkout', '-b', branch)
        git('-C', str(checkout), 'config', 'user.name', user['login'])
        git('-C', str(checkout), 'config', 'user.email', f"{user['id']}+{user['login']}@users.noreply.github.com")
        git('-C', str(checkout), 'add', '--', 'submissions/' + STUDENT)
        # The course repo ignores every README.md; force-stage the required student README.
        git('-C', str(checkout), 'add', '-f', '--', 'submissions/' + STUDENT + '/README.md')
        if not git('-C', str(checkout), 'diff', '--cached', '--name-only').strip():
            print('Khong co thay doi so voi nhanh goc; khong tao PR rong.')
            return None
        git('-C', str(checkout), 'commit', '-m', 'Submit DeepWeeds Day 2: ' + STUDENT)
        git('-C', str(checkout), 'push', url, 'HEAD:refs/heads/' + branch)
    print('Push thanh cong:', branch)
    try:
        pr = github_api('repos/' + REPO + '/pulls', token, {
            'title': 'Lab Day 2 — Vu Gia Khai — 2A202602786', 'head': branch,
            'base': info['default_branch'], 'draft': True,
            'body': 'Day 2 submission in submissions/' + STUDENT +
                    '. Includes workbook, generated report awaiting review, code, curves, predictions and reproducibility logs. '
                    'Please review experimental evidence before merging.'})
        print('Draft PR:', pr['html_url'])
        return pr['html_url']
    except urllib.error.HTTPError as error:
        print('Push da xong, tao PR that bai (HTTP', error.code, '). Kiem tra Pull requests: Read and write.')
        print('Tao PR thu cong:', 'https://github.com/' + REPO + '/compare/' + info['default_branch'] + '...' + branch + '?expand=1')
        return None
