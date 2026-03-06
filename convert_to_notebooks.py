# 將所有 phase-*/??_*.py 轉換成 Jupyter Notebook (.ipynb)
# - 頂部 docstring -> Markdown cell
# - # ==== 分隔的段落 -> 各自的 code cell

import json
import re
import glob
import os


def make_cell(cell_type, source):
    """建立一個 notebook cell"""
    lines = source if isinstance(source, list) else source.split('\n')
    # 確保每行結尾有 \n（除了最後一行）
    src = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            src.append(line + '\n' if not line.endswith('\n') else line)
        else:
            src.append(line.rstrip('\n'))
    # 移除尾部空行
    while src and src[-1].strip() == '':
        src.pop()
    if not src:
        return None

    if cell_type == 'markdown':
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": src
        }
    else:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src
        }


def extract_docstring(lines):
    """提取檔案頂部的 docstring，回傳 (docstring_lines, remaining_lines)"""
    content = '\n'.join(lines)

    # 找 triple-quote docstring
    match = re.match(r'^("""(.*?)""")', content, re.DOTALL)
    if not match:
        match = re.match(r"^('''(.*?)''')", content, re.DOTALL)
    if not match:
        return None, lines

    docstring_text = match.group(2).strip()
    end_pos = match.end(1)
    remaining = content[end_pos:].lstrip('\n')
    remaining_lines = remaining.split('\n') if remaining else []

    return docstring_text, remaining_lines


def split_into_sections(lines):
    """
    根據 # ===== 分隔線把程式碼切成段落
    """
    sections = []
    current = []

    for line in lines:
        # 偵測分隔線 (至少 10 個 =)
        if re.match(r'^# ={10,}', line):
            # 把之前累積的存起來
            if current:
                text = '\n'.join(current).strip()
                if text:
                    sections.append(current.copy())
            current = [line]
        else:
            current.append(line)

    if current:
        text = '\n'.join(current).strip()
        if text:
            sections.append(current)

    return sections


def section_to_cells(section_lines):
    """
    將一個 section 轉成 cell(s)
    - 如果開頭是 # ==== 的標題區，提取標題做 markdown
    - 剩下的是 code
    """
    cells = []
    code_lines = []
    i = 0

    # 看開頭是否有 # ==== 標題區塊
    header_lines = []
    while i < len(section_lines) and re.match(r'^# ={10,}', section_lines[i]):
        i += 1
    # 收集標題行（# 開頭的行，在 === 之後）
    while i < len(section_lines):
        line = section_lines[i]
        if re.match(r'^# ={10,}', line):
            i += 1
            continue
        if line.startswith('# ') and not line.startswith('# ---'):
            header_lines.append(line)
            i += 1
        else:
            break

    # 如果有標題，做成 markdown cell
    if header_lines:
        md_lines = []
        for hl in header_lines:
            text = hl.lstrip('# ').strip()
            if not md_lines:
                md_lines.append(f'## {text}')
            else:
                md_lines.append(text)
        cells.append(make_cell('markdown', '\n'.join(md_lines)))

    # 剩下的都是 code
    remaining = section_lines[i:]
    # 移除開頭和結尾的空行
    while remaining and remaining[0].strip() == '':
        remaining.pop(0)
    while remaining and remaining[-1].strip() == '':
        remaining.pop()

    if remaining:
        cells.append(make_cell('code', '\n'.join(remaining)))

    return [c for c in cells if c is not None]


def convert_py_to_notebook(py_path):
    """將一個 .py 檔轉成 .ipynb"""
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    cells = []

    # 1. 提取頂部 docstring → markdown
    docstring, remaining = extract_docstring(lines)
    if docstring:
        # 轉換 docstring 格式
        md_lines = []
        for line in docstring.split('\n'):
            # ===== 分隔線 → markdown ---
            if re.match(r'^={5,}', line):
                md_lines.append('---')
            elif re.match(r'^─{5,}', line):
                md_lines.append('')
            else:
                md_lines.append(line)
        cells.append(make_cell('markdown', '\n'.join(md_lines)))

    # 2. 加 import cell（收集所有 import 行）
    import_lines = []
    non_import_lines = []
    in_imports = True
    for line in remaining:
        stripped = line.strip()
        if in_imports and (stripped.startswith('import ') or
                           stripped.startswith('from ') or
                           stripped == '' or
                           stripped.startswith('#')):
            import_lines.append(line)
        else:
            in_imports = False
            non_import_lines.append(line)

    # 清理 import 區塊
    while import_lines and import_lines[-1].strip() == '':
        import_lines.pop()
    while import_lines and import_lines[0].strip() == '':
        import_lines.pop(0)

    if import_lines:
        cells.append(make_cell('code', '\n'.join(import_lines)))

    # 3. 按 # ==== 切段落
    sections = split_into_sections(non_import_lines)

    for section in sections:
        new_cells = section_to_cells(section)
        cells.extend(new_cells)

    # 移除 None cells
    cells = [c for c in cells if c is not None]

    # 建立 notebook
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.3"
            }
        },
        "cells": cells
    }

    # 寫入
    ipynb_path = py_path.replace('.py', '.ipynb')
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    return ipynb_path


# 主程式
if __name__ == '__main__':
    py_files = sorted(glob.glob('phase-*/[0-9]*.py'))
    print(f"找到 {len(py_files)} 個 .py 檔案\n")

    for py_file in py_files:
        ipynb_path = convert_py_to_notebook(py_file)
        # 計算 cell 數
        with open(ipynb_path, 'r') as f:
            nb = json.load(f)
        n_cells = len(nb['cells'])
        n_md = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
        n_code = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
        print(f"  {py_file:55s} → {n_cells:2d} cells ({n_md} md + {n_code} code)")

    print(f"\n轉換完成！共 {len(py_files)} 個 notebook")
