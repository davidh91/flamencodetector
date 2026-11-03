#!/usr/bin/env bash
# Fix files that macOS lists but cannot open on an exFAT drive.
#
# Every exFAT file entry stores a checksum of its name (the "name hash").
# macOS uses it to look up a file; a wrong hash means "No such file", even
# though `ls` shows the file. All broken files have accented names. A rename
# on Windows writes a fresh entry with a correct hash, so we rename each file
# with a non-ASCII name to a temporary name and back. The name stays the same.
#
# Run in Git Bash on Windows:
#   bash fix_names.sh /e/<collection>          # dry run: only lists the files
#   bash fix_names.sh /e/<collection> --apply  # really renames

set -euo pipefail

root="${1:?usage: bash fix_names.sh /e/<collection> [--apply]}"
apply=0
[[ "${2:-}" == "--apply" ]] && apply=1

if [[ ! -d "$root" ]]; then
  echo "Folder not found: $root" >&2
  exit 1
fi

echo "Scanning $root ..."
count=0
while IFS= read -r -d '' file; do
  name="$(basename "$file")"
  # Byte-wise check (LC_ALL=C): any byte outside printable ASCII means an accent.
  if ! LC_ALL=C grep -q '[^ -~]' <<<"$name"; then
    continue
  fi
  count=$((count + 1))
  if (( apply )); then
    dir="$(dirname "$file")"
    temp="$dir/__fix_$$_$count.tmp"
    mv -- "$file" "$temp"
    mv -- "$temp" "$file"
    echo "rewritten:  $file"
  else
    echo "to rewrite: $file"
  fi
done < <(find "$root" -type f -print0)

if (( apply )); then
  echo "$count file(s) rewritten."
else
  echo "$count file(s) with accents. Run again with --apply to rewrite them."
fi
