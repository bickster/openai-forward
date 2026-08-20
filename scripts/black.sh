#!/bin/bash
pip install black==22.3.0
arrVar=()
echo we ignore non-*.py files
excluded_files=(
)
for changed_file in $CHANGED_FILES; do
  if [[ ${changed_file} == *.py ]] && ! [[ " ${excluded_files[@]} " =~ " ${changed_file} " ]]; then
    echo checking ${changed_file}
    arrVar+=(${changed_file})
  fi
done
if (( ${#arrVar[@]} )); then
  # Propagate black's exit status. Without this the trailing `exit 0` swallowed it and the
  # check-black job passed no matter what black found.
  black -S --check "${arrVar[@]}" || exit 1
fi
echo "no files left to check"
exit 0