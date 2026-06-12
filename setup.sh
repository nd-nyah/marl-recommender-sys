#!/bin/bash
set -e

echo "======================================"
echo "🚀 marl-recommender setup"
echo "======================================"

ACTION=$1

# -----------------------------
# Validate input
# -----------------------------
if [[ -z "$ACTION" ]]; then
    echo "❌ No flag provided"
    echo ""
    echo "Usage:"
    echo "  bash setup.sh --venv"
    echo "  bash setup.sh --data"
    echo "  bash setup.sh --both"
    exit 1
fi

# -----------------------------
# FUNCTION: VENV SETUP
# -----------------------------
setup_venv () {

    echo "📦 Creating virtual environment..."

    rm -rf .venv
    python3 -m venv .venv

    echo "📦 Activating virtual environment..."
    source .venv/bin/activate

    echo "⬆️ Bootstrapping pip inside venv..."
    python3 -m ensurepip --upgrade || true

    echo "⬆️ Upgrading pip + tools..."
    python3 -m pip install --upgrade pip setuptools wheel

    echo "📚 Installing project (editable mode)..."
    python3 -m pip install -e . || echo "⚠️ Project install failed (check pyproject.toml or setup.cfg)"

    echo "✅ Virtual environment ready"
}

# -----------------------------
# FUNCTION: DATA DOWNLOAD
# -----------------------------
download_data () {

    echo "⬇️ Downloading MovieLens dataset..."

    mkdir -p data/raw
    mkdir -p data/movielens

    MOVIELENS_URL="https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

    ZIP_FILE="data/raw/movielens.zip"
    EXTRACT_DIR="data/movielens"

    # Download if not exists
    if [ ! -f "$ZIP_FILE" ]; then
        echo "⬇️ Fetching MovieLens..."
        curl -L "$MOVIELENS_URL" -o "$ZIP_FILE"
    else
        echo "📦 MovieLens zip already exists"
    fi

    echo "📂 Extracting dataset..."

    if command -v unzip >/dev/null 2>&1; then
        unzip -o "$ZIP_FILE" -d "$EXTRACT_DIR"
    else
        echo "⚠️ unzip not found → using Python fallback"

        python3 - <<EOF
import zipfile

zip_path = "$ZIP_FILE"
out_dir = "$EXTRACT_DIR"

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(out_dir)

print("✅ Extraction complete")
EOF
    fi

    RAW_DIR="$EXTRACT_DIR/ml-latest-small"

    # Validate dataset
    if [ -f "$RAW_DIR/ratings.csv" ] && [ -f "$RAW_DIR/movies.csv" ]; then
        echo "✅ MovieLens dataset ready"
        echo "📁 Location: $RAW_DIR"
        ls -lh "$RAW_DIR"
    else
        echo "❌ Dataset extraction failed"
        exit 1
    fi
}

# -----------------------------
# FUNCTION: FOLDERS
# -----------------------------
create_folders () {

    echo "📁 Creating project folders..."

    mkdir -p data/movielens
    mkdir -p data/processed
    mkdir -p checkpoints
    mkdir -p logs
    mkdir -p results
    mkdir -p notebooks
    mkdir -p tests

    mkdir -p src/marl_recommender/{envs,agents,algorithms,metrics,evaluation,training,utils,common}

    touch src/marl_recommender/__init__.py

    echo "📚 Installing project (editable mode)..."
    pip install -e . || echo "⚠️ Skipping install (run after venv setup)"
}

# -----------------------------
# ROUTER
# -----------------------------
if [[ "$ACTION" == "--venv" ]]; then
    setup_venv

elif [[ "$ACTION" == "--data" ]]; then
    download_data

elif [[ "$ACTION" == "--both" ]]; then
    setup_venv
    create_folders
    download_data

else
    echo "❌ Invalid option: $ACTION"
    exit 1
fi

# -----------------------------
# DONE
# -----------------------------
echo "======================================"
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python -m src.marl_recommender"
echo "======================================"
