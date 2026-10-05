"""Check GitHub credentials without training or creating a branch/PR."""
from pathlib import Path
import os
import subprocess
import tempfile
from submission import get_token, github_api, REPO


def check():
    token = get_token()
    if not token:
        print('GITHUB_TOKEN: MISSING. Kaggle Add-ons > Secrets > GITHUB_TOKEN.')
        print('Training and ZIP export can run; Git push/PR cannot authenticate yet.')
        return False
    try:
        info = github_api('repos/' + REPO, token)
        print('Repository:', info['full_name'], '| base:', info['default_branch'])
        if not info.get('permissions', {}).get('push'):
            print('FAIL: this GitHub account/token has no write permission to the repo.')
            return False
        with tempfile.TemporaryDirectory() as tmp:
            askpass = Path(tmp) / 'askpass.py'
            askpass.write_text('#!/usr/bin/env python3\nimport os,sys\nprint("x-access-token" if "username" in sys.argv[1].lower() else os.environ["DAY2_GIT_TOKEN"])\n')
            askpass.chmod(0o700)
            env = dict(os.environ, DAY2_GIT_TOKEN=token, GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT='0')
            result = subprocess.run(['git', 'push', '--dry-run', 'https://github.com/' + REPO + '.git',
                                     'HEAD:refs/heads/results/kaggle-preflight'], env=env,
                                    capture_output=True, text=True)
            print('Git push dry-run:', 'PASS' if result.returncode == 0 else 'FAIL')
            if result.returncode:
                print((result.stderr or result.stdout).replace(token, '[REDACTED]'))
                return False
        print('No branch or PR was created. PR creation also needs Pull requests: Read and write.')
        return True
    except Exception as error:
        print('FAIL:', type(error).__name__, str(error).replace(token, '[REDACTED]'))
        return False
