import os

path = "/Users/gouravsstudy/Desktop/AI Revision, and Fun Learning/Algorithm/package_upsell_model.py"

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Make it an interactive script
blocks = content.split('"""\n')
new_content = ""

for i, block in enumerate(blocks):
    if i == 0:
        new_content += "# %%\n" + block
    elif i % 2 == 1:
        # Markdown start
        new_content += "# %%\n\"\"\"\n" + block
    else:
        # Code start
        new_content += '"""\n# %%\n' + block

with open(path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Formatted as interactive cells.")
