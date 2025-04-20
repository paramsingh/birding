import os
import pandas as pd
from pathlib import Path
import librosa
import soundfile as sf # soundfile might be needed if librosa uses it under the hood or for specific formats
from collections import Counter
from tqdm.auto import tqdm # Use auto for notebook/console compatibility
import numpy as np

# Define common audio formats
AUDIO_FORMATS = [".ogg", ".wav", ".mp3", ".flac", ".aiff", ".aif"]

# --- Task 0.2.1: List all audio files ---
def list_audio_files(train_audio_path):
    """Lists all audio files recursively within the training directory."""
    audio_files = []
    path_obj = Path(train_audio_path)
    for item in path_obj.rglob('*'):
        if item.is_file() and item.suffix.lower() in AUDIO_FORMATS:
            audio_files.append(str(item))
    return audio_files

# --- Task 0.2.2: Count files per group ---
def count_files_per_group(audio_files, taxonomy_df):
    """Counts files per species and taxonomic group (class)."""
    species_labels = [Path(f).parent.name for f in audio_files]
    species_counts = Counter(species_labels)

    # Map species to class using taxonomy
    species_to_class = taxonomy_df.set_index('primary_label')['class'].to_dict()

    group_counts = Counter()
    for species, count in species_counts.items():
        class_name = species_to_class.get(species, "Unknown") # Handle missing species in taxonomy
        group_counts[class_name] += count

    return dict(species_counts), dict(group_counts)

# --- Task 0.2.3: Extract file metadata ---
def extract_file_metadata(file_path):
    """Extracts duration, sampling rate, and format for a single audio file."""
    metadata = {
        'file_path': file_path,
        'duration': None,
        'sampling_rate': None,
        'format': Path(file_path).suffix.lower(),
        'error': None
    }
    if metadata['format'] not in AUDIO_FORMATS:
        metadata['error'] = f"Non-audio extension: {metadata['format']}"
        return metadata

    try:
        # Use soundfile first for potentially better format support and speed
        # It might fail for some formats librosa handles, hence the fallback
        try:
            with sf.SoundFile(file_path) as f:
                metadata['duration'] = f.frames / f.samplerate
                metadata['sampling_rate'] = f.samplerate
        except Exception as sf_err:
            # Fallback to librosa if soundfile fails
            y, sr = librosa.load(file_path, sr=None, mono=False) # Load with original sr
            metadata['sampling_rate'] = sr
            metadata['duration'] = librosa.get_duration(y=y, sr=sr)

    except Exception as e:
        metadata['error'] = f"Error loading/processing file: {str(e)}"

    return metadata


# --- Task 0.2.4: Create metadata DataFrame ---
def create_metadata_dataframe(metadata_list, taxonomy_df):
    """Creates a pandas DataFrame from the extracted metadata list and merges taxonomy info."""
    metadata_df = pd.DataFrame(metadata_list)

    # Extract primary_label from file_path (assuming parent directory is the label)
    metadata_df['primary_label'] = metadata_df['file_path'].apply(lambda x: Path(x).parent.name)

    # Merge with taxonomy information
    metadata_df = pd.merge(
        metadata_df,
        taxonomy_df[['primary_label', 'species_code', 'genus', 'family', 'order', 'class']],
        on='primary_label',
        how='left' # Keep all files even if species not in taxonomy
    )
    return metadata_df

# --- Task 0.2.5: Generate summary statistics ---
def generate_summary_statistics(metadata_df):
    """Generates a dictionary of summary statistics from the metadata DataFrame."""
    summary = {}

    summary['total_files'] = len(metadata_df)
    summary['total_species'] = metadata_df['primary_label'].nunique()
    summary['total_taxonomic_groups'] = metadata_df['class'].nunique()

    summary['files_per_species'] = metadata_df['primary_label'].value_counts().to_dict()
    summary['files_per_group'] = metadata_df['class'].value_counts().to_dict()

    # Duration stats (handle potential NaNs)
    valid_durations = metadata_df['duration'].dropna()
    summary['duration_stats'] = {
        'mean': valid_durations.mean(),
        'median': valid_durations.median(),
        'min': valid_durations.min(),
        'max': valid_durations.max(),
        'std': valid_durations.std(),
        'total_hours': valid_durations.sum() / 3600
    }

    # Sampling rate and format counts (handle potential NaNs)
    summary['sampling_rate_stats'] = metadata_df['sampling_rate'].dropna().value_counts().to_dict()
    summary['format_counts'] = metadata_df['format'].value_counts().to_dict()

    # Error count
    summary['error_count'] = metadata_df['error'].notna().sum()

    return summary


# --- Main Execution Logic (Optional: for running as a script) ---
def perform_data_inventory(data_dir):
    """Performs all data inventory tasks sequentially."""
    data_dir = Path(data_dir)
    train_audio_path = data_dir / "train_audio"
    taxonomy_path = data_dir / "taxonomy.csv"

    print("Verifying data access...")
    if not train_audio_path.exists() or not taxonomy_path.exists():
        print(f"Error: Required data not found in {data_dir}")
        return None
    print("Data access verified.")

    print("Loading taxonomy...")
    taxonomy_df = pd.read_csv(taxonomy_path)
    print(f"Taxonomy loaded: {len(taxonomy_df)} species.")

    print("Listing audio files...")
    audio_files = list_audio_files(train_audio_path)
    if not audio_files:
        print("Error: No audio files found.")
        return None
    print(f"Found {len(audio_files)} audio files.")

    print("Counting files per group...")
    species_counts, group_counts = count_files_per_group(audio_files, taxonomy_df)
    print("Counts calculated.")

    print("Extracting metadata (this may take a while)...")
    metadata_list = [extract_file_metadata(f) for f in tqdm(audio_files)]
    print("Metadata extraction complete.")

    print("Creating metadata DataFrame...")
    metadata_df = create_metadata_dataframe(metadata_list, taxonomy_df)
    print("DataFrame created.")

    print("Generating summary statistics...")
    summary_stats = generate_summary_statistics(metadata_df)
    print("Summary statistics generated.")

    # Optionally save or return results
    # metadata_df.to_csv(data_dir / "metadata_inventory.csv", index=False)
    # with open(data_dir / "summary_statistics.json", 'w') as f:
    #     import json
    #     json.dump(summary_stats, f, indent=4)

    return metadata_df, summary_stats

if __name__ == "__main__":
    # Example usage when run as a script
    # Assumes the script is run from a directory where './data' exists
    DEFAULT_DATA_DIR = "../data" # Adjust if your structure is different
    metadata_df, summary_stats = perform_data_inventory(DEFAULT_DATA_DIR)

    if metadata_df is not None and summary_stats is not None:
        print("\n--- Data Inventory Summary ---")
        print(f"Total Files: {summary_stats['total_files']}")
        print(f"Total Species: {summary_stats['total_species']}")
        print(f"Total Taxonomic Groups: {summary_stats['total_taxonomic_groups']}")
        print(f"Errors Encountered: {summary_stats['error_count']}")
        print(f"Total Audio Duration: {summary_stats['duration_stats']['total_hours']:.2f} hours")
        print("\nSampling Rates:")
        for rate, count in summary_stats['sampling_rate_stats'].items():
            print(f"  - {int(rate)} Hz: {count} files")
        print("\nFormats:")
        for fmt, count in summary_stats['format_counts'].items():
            print(f"  - {fmt}: {count} files")

        # Display species with fewest files
        print("\nSpecies with Fewest Files:")
        sorted_species = sorted(summary_stats['files_per_species'].items(), key=lambda item: item[1])
        for species, count in sorted_species[:10]: # Show top 10 rarest
            print(f"  - {species}: {count} files")
