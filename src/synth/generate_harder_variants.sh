#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default values
num_variants=3
pattern="*"
usage_msg="Usage: $0 [<num_variants>] [<pattern>] | $0 [<pattern>] [<num_variants>] | $0 [<pattern_with_quotes>]"

# Argument parsing
arg1="$1"
arg2="$2"

is_arg1_numeric=false
is_arg2_numeric=false
if [[ "$arg1" =~ ^[0-9]+$ ]]; then is_arg1_numeric=true; fi
if [[ "$arg2" =~ ^[0-9]+$ ]]; then is_arg2_numeric=true; fi

if [ $# -eq 0 ]; then
    # Use defaults
    :
elif [ $# -eq 1 ]; then
    if $is_arg1_numeric; then
        num_variants=$arg1
    else
        pattern=$arg1
    fi
elif [ $# -eq 2 ]; then
    if $is_arg1_numeric && ! $is_arg2_numeric; then
        # num_variants pattern
        num_variants=$arg1
        pattern=$arg2
    elif ! $is_arg1_numeric && $is_arg2_numeric; then
        # pattern num_variants
        pattern=$arg1
        num_variants=$arg2
    elif $is_arg1_numeric && $is_arg2_numeric; then
         echo "Error: Cannot determine arguments. Both arguments are numeric." >&2
         echo "$usage_msg" >&2
         exit 1
    else # Neither is numeric
         echo "Error: Cannot determine arguments. Expected one numeric <num_variants> and one string <pattern>." >&2
         echo "$usage_msg" >&2
         exit 1
    fi
elif [ $# -gt 2 ]; then
     echo "Error: Too many arguments ($# received). Did you forget to quote the pattern?" >&2
     echo "Example: $0 3 '*_variant_1*'" >&2
     echo "$usage_msg" >&2
     exit 1
fi

echo "Using pattern: '$pattern' to match JSON files"
echo "Number of variants to generate: $num_variants"

# Define the target directory
target_dir="${SCRIPT_DIR}/synth_network_specs"

# First, find and print all matching files using find
echo "Searching for files in: ${target_dir}"
echo "Matching files with pattern: ${pattern}.json"

matching_files=()
# Use null delimiter correctly with find -print0
while IFS= read -r -d $'\0' file; do
    # Check if find actually outputted something (handles edge cases)
    if [[ -n "$file" ]]; then
        matching_files+=("$file")
    fi
done < <(find "${target_dir}" -maxdepth 1 -name "${pattern}.json" -print0)

# Check if any files matched
if [ ${#matching_files[@]} -eq 0 ]; then
    echo "No files matched the pattern '${pattern}.json' in ${target_dir}/"
    exit 1
fi

# Print matching files (basenames only)
echo "Matching files found:"
for file in "${matching_files[@]}"; do
    echo "  - $(basename "$file")"
done

echo "Found ${#matching_files[@]} matching files to process."
echo "Processing files..."

# Process the matched files
for json_file in "${matching_files[@]}"; do
    echo "Processing $json_file..."
    # Run the Python script with the input-spec flag
    python "${SCRIPT_DIR}/network_instance_synthesizer.py" --input-spec "$json_file" --num-variants "$num_variants"
    
    # Check if command was successful
    if [ $? -ne 0 ]; then
        echo "Error processing $json_file"
    fi
done

echo "All files processed!"
