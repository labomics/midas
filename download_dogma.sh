# download
FILE="GSE166188_RAW.tar"
url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE166nnn/GSE166188/suppl/GSE166188_RAW.tar"
if [ -f "$FILE" ]; then
    echo "$FILE already exists. Skipping download."
else
    echo "$FILE does not exist. Downloading..."
    wget "$url" -O "$FILE"
fi

# decompression
mkdir ./data/raw/atac+rna+adt/dogma
tar -xvf GSE166188_RAW.tar -C ./data/raw/atac+rna+adt/dogma
cd ./data/raw/atac+rna+adt/dogma

# categorize
for file in *; do
    if [[ -f "$file" ]]; then
        dir_name=$(echo "$file" | awk -F'_' '{print $2"_"$3}' | tr '[:upper:]' '[:lower:]')
        if [[ ! -d "$dir_name" ]]; then
            mkdir "$dir_name"
        fi
        mv "$file" "$dir_name/"
    fi
done

# 
for dir in *; do
    for file in "$dir"/*_ATAC_*.gz; do
        if [[ -f "$file" ]]; then
            prefix=$(echo "$file" | awk -F'_' '{print $1"_"$2"_"$3"_"$4"_"$5"_"$6"_"$7}')_
            prefix2=$(echo "$file" | awk -F'_' '{print $1"_"$2"_"$3"_"$4"_"$5"_"$6"_"$7}')
            actual_name="${file#$prefix}"
            if [[ ! -d "$prefix" ]]; then
                mkdir "$prefix2"
            fi
            mv "$file" "$prefix2/$actual_name"
        fi
    done

done