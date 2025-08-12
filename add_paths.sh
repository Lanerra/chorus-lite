#!/bin/bash

# This script adds the relative path of each Python file as a comment at the top.

find . -type f -name "*.py" -print0 | while IFS= read -r -d $'\0' file; do
  # Remove the leading './' from the file path for a cleaner look.
  relative_path="${file#./}"

  # This is the comment that will be added to the top of the file.
  comment="# ${relative_path}"

  # Read the first line of the file to check if the comment already exists.
  first_line=$(head -n 1 "$file" 2>/dev/null)

  # If the comment isn't already there, add it.
  if [[ "$first_line" != "$comment" ]]; then
    sed -i "1i ${comment}" "$file"
  fi
done

echo "Done! Relative paths have been added to all Python files. ✅"
