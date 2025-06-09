#!/usr/bin/env python3
"""
format_all.py

プロジェクト直下で実行すると、
- Python (*.py) ファイルを black でフォーマット
- Markdown (*.md) ファイルを mdformat でフォーマット
を再帰的に行います。
"""

import sys
import subprocess
from pathlib import Path

# フォーマット対象のディレクトリ・ファイル名
TARGETS = [
    "README.md",
    "streamlit/pages"
]

def run(cmd):
    print(f"> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    root = Path(__file__).parent.resolve()

    # Python ファイルを black で
    print("▶ Python ファイルを black で整形中…")
    for target in TARGETS:
        path = root / target
        if path.exists():
            # black はファイル・ディレクトリどちらも受け付けます
            run([sys.executable, "-m", "black", str(path)])
        else:
            print(f"⚠️  存在しません: {path}")

    # Markdown ファイルを mdformat で
    print("\n▶ Markdown ファイルを mdformat で整形中…")
    for target in TARGETS:
        path = root / target
        if path.exists():
            run([sys.executable, "-m", "mdformat", str(path)])
        else:
            # 存在しない場合はスキップ
            continue

    print("\n✅ 全ファイルのフォーマットが完了しました。")

if __name__ == "__main__":
    main()
