import json
from pathlib import Path

nb_path = Path("Colab_DeepWeeds.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        # Support Kaggle Secrets in Cell 10
        if "=== TỰ ĐỘNG COMMIT, PUSH & TẠO PULL REQUEST (PR) ===" in src:
            old_str = """# 1. Tự động đọc từ Colab Secrets (nếu đã lưu hình chìa khóa 🔑 bên trái menu Colab)
try:
    from google.colab import userdata
    gh_token = userdata.get('GITHUB_TOKEN')
    if gh_token:
        print("🔑 Đã tự động nhận diện GITHUB_TOKEN từ Colab Secrets (không cần nhập tay)!")
except Exception:
    pass"""
            new_str = """# 1. Tự động đọc từ Colab Secrets hoặc Kaggle Secrets (nếu đã lưu)
try:
    from google.colab import userdata
    gh_token = userdata.get('GITHUB_TOKEN')
    if gh_token:
        print("🔑 Đã tự động nhận diện GITHUB_TOKEN từ Colab Secrets (không cần nhập tay)!")
except Exception:
    pass

if not gh_token:
    try:
        from kaggle_secrets import UserSecretsClient
        gh_token = UserSecretsClient().get_secret('GITHUB_TOKEN')
        if gh_token:
            print("🔑 Đã tự động nhận diện GITHUB_TOKEN từ Kaggle Secrets (Add-ons -> Secrets)!")
    except Exception:
        pass"""
            if old_str in src:
                src = src.replace(old_str, new_str)
                cell["source"] = [line + "\n" for line in src.splitlines()]

        # Ensure Kaggle Output path in Cell 11
        if "deepweeds_lab2_results_" in src and "zipfile.ZipFile" in src:
            if "if IN_KAGGLE:" not in src:
                src = src.replace(
                    "zip_name = f'deepweeds_lab2_results_{timestamp}.zip'",
                    "zip_name = f'/kaggle/working/deepweeds_lab2_results_{timestamp}.zip' if os.path.exists('/kaggle/working') else f'deepweeds_lab2_results_{timestamp}.zip'"
                )
                cell["source"] = [line + "\n" for line in src.splitlines()]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Updated Kaggle support in Colab_DeepWeeds.ipynb.")
