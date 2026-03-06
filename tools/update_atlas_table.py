"""
Sync the atlas table in README.md from the BrainGlobe docs repo.

Run manually when _atlas_table.md changes.
Usage: python tools/update_atlas_table.py.
"""

import urllib.request

# Where to get the table from
url = (
    "https://raw.githubusercontent.com/brainglobe/brainglobe.github.io/"
    "main/docs/source/documentation/brainglobe-atlasapi/_atlas_table.md"
)

# Markers in README.md
start_marker = "<!-- BEGIN_ATLAS_TABLE -->"
end_marker = "<!-- END_ATLAS_TABLE -->"

# Get the table from the docs repo
print("Fetching table from docs repo...")
response = urllib.request.urlopen(url, timeout=10)
table_content = response.read().decode("utf-8").strip()
response.close()

# Read the README.md file
readme_path = "README.md"
print(f"Reading {readme_path}...")
with open(readme_path, "r", encoding="utf-8") as f:
    readme_text = f.read()

# Check that markers exist
if start_marker not in readme_text:
    print(f"ERROR: Could not find '{start_marker}' in README.md")
    exit(1)

if end_marker not in readme_text:
    print(f"ERROR: Could not find '{end_marker}' in README.md")
    exit(1)

# Split the README into parts
before = readme_text.split(start_marker)[0]
after = readme_text.split(end_marker)[1]

# Put it back together with the new table
new_readme = (
    before
    + start_marker
    + "\n\n"
    + table_content
    + "\n\n"
    + end_marker
    + after
)

# Write it back
print(f"Writing updated table to {readme_path}...")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(new_readme)

print("Done!")
