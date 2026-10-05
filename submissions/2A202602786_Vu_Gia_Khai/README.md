# Lab Day 2 — Vu Gia Khai — 2A202602786

Notebook: https://www.kaggle.com/code/b22dckh065vgiakhi/notebooke997c35844

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
