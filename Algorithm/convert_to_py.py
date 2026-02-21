import json
import os

def convert(notebook_path):
    py_path = notebook_path.replace('.ipynb', '.py')
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    with open(py_path, 'w', encoding='utf-8') as out:
        out.write("# Predict Package Upsells (Flight + Hotel)\n\n")
        for cell in nb['cells']:
            if cell['cell_type'] == 'markdown':
                out.write('"""\n')
                out.writelines(cell.get('source', []))
                out.write('\n"""\n\n')
            elif cell['cell_type'] == 'code':
                out.writelines(cell.get('source', []))
                out.write('\n\n')
                
    print(f"Created script at {py_path}")

convert("/Users/gouravsstudy/Desktop/AI Revision, and Fun Learning/Algorithm/package_upsell_model.ipynb")
