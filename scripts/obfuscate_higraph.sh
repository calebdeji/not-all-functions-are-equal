#!/bin/bash
#
# USAGE:
#   bash obfuscate_higraph.sh                   # all years
#   bash obfuscate_higraph.sh 2012 2016         # 2012 to 2016 inclusive
#   bash obfuscate_higraph.sh 2017 2019         # 2017 to 2019 inclusive

# Staging directory containing only novel-family APKs.
# Run src/data/prepare_novel_apks.py first to populate this.
INPUT_DIR="raw_format/higraph_novel"

YEAR_FROM=${1:-0}
YEAR_TO=${2:-9999}

count=0

# Check if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory '$INPUT_DIR' not found!"
    exit 1
fi

echo "Year range: ${1:-all} — ${2:-all}"

# Define the obfuscation configurations.
# Format: "output_folder_name|obfuscator_flags"
# You can add or modify these based on the specific techniques you want to test.
CONFIGS=(
    "arithmetic_branch|-o ArithmeticBranch"
    "call_indirection|-o CallIndirection"
    "reorder|-o Reorder"
)

# Loop through the year directories (e.g., 2012, 2013)
for year_dir in "$INPUT_DIR"/*; do
    
    # Skip if it is not a directory
    if [ ! -d "$year_dir" ]; then
        continue
    fi
    
    # Extract just the year (e.g., "2012")
    year=$(basename "$year_dir")

    if [[ "$year" -lt "$YEAR_FROM" ]] || [[ "$year" -gt "$YEAR_TO" ]]; then
      continue
    fi

    # Collect APKs for this year so we know the total upfront
    mapfile -t year_apks < <(find "$year_dir" -maxdepth 1 -name "*.apk" | sort)
    year_total=${#year_apks[@]}
    if [ "$year_total" -eq 0 ]; then
        continue
    fi
    year_count=0

    # Loop through all APK files in the current year directory
    for apk_file in "${year_apks[@]}"; do
        year_count=$(( year_count + 1 ))

        echo "=================================================="
        echo "Processing sample ${year_count}/${year_total} [${year}]: $apk_file"
        echo "=================================================="
        
        apk_name=$(basename "$apk_file")

        # Iterate through each obfuscation setup for the current APK
        for config in "${CONFIGS[@]}"; do
            
            # Split the configuration string into the folder name and the flags
            IFS="|" read -r output_base flags <<< "$config"

            # Create the target directory structure (e.g., obfuscated_higraph/arithmetic_branch/2012/)
            target_dir="obfuscated_higraph/${output_base}/${year}"
            mkdir -p "$target_dir"

            # Define the final output path for the APK
            target_apk="${target_dir}/${apk_name}"

            # ---------------------------------------------------------
            # NEW CHECK: Skip if the output file already exists
            # ---------------------------------------------------------
            if [ -f "$target_apk" ]; then
                echo "  -> Skipping [$output_base] configuration... already processed."
                echo "  ------------------------------------------------"
                continue
            fi

            echo "  -> Applying [$output_base] configuration..."
            
            # Read the flags into an array so Docker interprets the spaces correctly
            read -a flag_array <<< "$flags"

            # Run the docker command
            # Note the -d flag is added to explicitly define where to save the output
            # We prepend /workdir/ to the paths because ${PWD} is mounted to /workdir
            docker run \
                --platform linux/amd64 \
                --rm \
                -u $(id -u):$(id -g) \
                -v "${PWD}":"/workdir" \
                obfuscapk \
                "${flag_array[@]}" \
                -o Rebuild \
                -o NewAlignment \
                -o NewSignature \
                -p \
                -d "/workdir/${target_apk}" \
                "/workdir/${apk_file}"
                
            echo "  -> Saved to: $target_apk"
            echo "  ------------------------------------------------"
        done

        echo "Finished all obfuscations for: $apk_name"
        echo ""

        count=$((count + 1))
    done
done

echo "All APKs processed successfully."