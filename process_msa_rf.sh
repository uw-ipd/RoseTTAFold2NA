#!/bin/bash

# Check if a character is a non-standard base, gap, or ambiguous base (N)
function is_non_standard_character() {
    local char=$1
    case "$char" in
        A|U|G|C|-|N)
            return 1 # Standard base, gap, or ambiguous base found
            ;;
        *)
            return 0 # Non-standard base, gap, or ambiguous base found
            ;;
    esac
}

# Check the input file and handle errors
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file.afa>"
    exit 1
fi

input_file="$1"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: The input file '$input_file' does not exist."
    exit 1
fi

# Extract the directory path from the input file's absolute path
input_dir=$(dirname "$input_file")

# Create a log file with the same name as the input file but with .log.txt extension
log_file="${input_dir}/$(basename "${input_file%.*}")_log.txt"

# Create a temporary file to store the new sequences in the same directory
temp_output_file="${input_dir}/$(basename "${input_file%.*}")_temp.afa"

# Redirect all output to the log file
exec > "$log_file"

# Initialize variables to store the current header and sequence
current_header=""
current_sequence=""

# Process the input file line by line
while IFS= read -r line; do
    if [[ "$line" =~ ^\> ]]; then
        # Process the previous sequence before moving to the new header
        if [ -n "$current_header" ] && [ -n "$current_sequence" ]; then
            # Check each character in the sequence for non-standard bases, gaps, or ambiguous bases
            skip_sequence=false
            for (( i=0; i<${#current_sequence}; i++ )); do
                char=${current_sequence:i:1}
                if is_non_standard_character "$char"; then
                    echo "Non-standard character '$char' found in the sequence:"
                    echo "$current_header"
                    echo "$current_sequence"
                    skip_sequence=true
                    break
                fi
            done

            # Write to the temporary output file if the sequence contains no non-standard characters
            if ! $skip_sequence; then
                echo "$current_header" >> "$temp_output_file"
                echo "$current_sequence" >> "$temp_output_file"
            fi
        fi

        # Save the new header and reset the current sequence
        current_header="$line"
        current_sequence=""
    else
        # Accumulate sequence lines under the same header
        current_sequence+="$line"
    fi
done < "$input_file"

# Process the last sequence after reaching the end of the file
if [ -n "$current_header" ] && [ -n "$current_sequence" ]; then
    # Check each character in the sequence for non-standard bases, gaps, or ambiguous bases
    skip_sequence=false
    for (( i=0; i<${#current_sequence}; i++ )); do
        char=${current_sequence:i:1}
        if is_non_standard_character "$char"; then
            echo "Non-standard character '$char' found in the sequence:"
            echo "$current_header"
            echo "$current_sequence"
            skip_sequence=true
            break
        fi
    done

    # Write to the temporary output file if the sequence contains no non-standard characters
    if ! $skip_sequence; then
        echo "$current_header" >> "$temp_output_file"
        echo "$current_sequence" >> "$temp_output_file"
    fi
fi

# Rename the old input file to basename_old.afa in the same directory
mv "$input_file" "${input_dir}/$(basename "${input_file%.*}")_old.afa"

# Rename the temporary file to the original input file in the same directory
mv "$temp_output_file" "$input_file"

# Provide a summary message in the log file
echo "Processed $input_file. Removed sequences with non-standard bases and saved the results in $input_file." >> "$log_file"
