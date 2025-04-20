import pytest
import pandas as pd
import os
from pathlib import Path
from unittest.mock import MagicMock # Import MagicMock
from data_inventory import (
    list_audio_files,
    count_files_per_group,
    extract_file_metadata,
    create_metadata_dataframe,
    generate_summary_statistics
)

# Define a fixture for a temporary data directory structure
@pytest.fixture(scope="module")
def temp_data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    train_audio = data_dir / "train_audio"
    train_audio.mkdir()
    taxonomy = pd.DataFrame({
        'primary_label': ['bird1', 'bird2', 'frog1', 'insect1'],
        'species_code': ['bir1', 'bir2', 'fro1', 'ins1'],
        'genus': ['GenusB', 'GenusB', 'GenusF', 'GenusI'],
        'family': ['FamilyB', 'FamilyB', 'FamilyF', 'FamilyI'],
        'order': ['OrderB', 'OrderB', 'OrderF', 'OrderI'],
        'class': ['Aves', 'Aves', 'Amphibia', 'Insecta']
    })
    taxonomy_path = data_dir / "taxonomy.csv"
    taxonomy.to_csv(taxonomy_path, index=False)

    # Create dummy audio files
    (train_audio / "bird1").mkdir()
    (train_audio / "bird2").mkdir()
    (train_audio / "frog1").mkdir()
    (train_audio / "insect1").mkdir()

    # Use different extensions and potentially invalid files
    (train_audio / "bird1" / "file1.ogg").touch()
    (train_audio / "bird1" / "file2.wav").touch() # Different extension
    (train_audio / "bird2" / "file3.ogg").touch()
    (train_audio / "frog1" / "file4.ogg").touch()
    (train_audio / "insect1" / "file5.ogg").touch()
    (train_audio / "bird1" / "not_audio.txt").touch() # Non-audio file
    (train_audio / "corrupted_dir").mkdir() # Directory instead of file path

    # Add a dummy ogg file with some content to test duration/sr extraction
    # Note: Actual audio libraries are needed for real metadata extraction
    # For testing purposes, we might mock these functions or use simplified logic
    sample_ogg_path = train_audio / "bird1" / "file1.ogg"
    with open(sample_ogg_path, 'wb') as f:
         # A minimal Ogg header might be needed for some tools, but mocking is safer
         f.write(b'OggS') # Dummy content

    return data_dir

# ---- Test Task 0.2.1: List all audio files ----
def test_list_audio_files(temp_data_dir):
    train_audio_path = temp_data_dir / "train_audio"
    expected_files = sorted([
        str(train_audio_path / "bird1" / "file1.ogg"),
        str(train_audio_path / "bird1" / "file2.wav"),
        str(train_audio_path / "bird2" / "file3.ogg"),
        str(train_audio_path / "frog1" / "file4.ogg"),
        str(train_audio_path / "insect1" / "file5.ogg"),
    ])
    # Assuming list_audio_files handles various extensions and ignores non-audio
    found_files = sorted(list_audio_files(str(train_audio_path)))
    assert found_files == expected_files
    assert str(train_audio_path / "bird1" / "not_audio.txt") not in found_files

# ---- Test Task 0.2.2: Count files per group ----
def test_count_files_per_group(temp_data_dir):
    train_audio_path = temp_data_dir / "train_audio"
    taxonomy_path = temp_data_dir / "taxonomy.csv"
    taxonomy_df = pd.read_csv(taxonomy_path)
    audio_files = list_audio_files(str(train_audio_path))

    species_counts, group_counts = count_files_per_group(audio_files, taxonomy_df)

    expected_species = {'bird1': 2, 'bird2': 1, 'frog1': 1, 'insect1': 1}
    expected_groups = {'Aves': 3, 'Amphibia': 1, 'Insecta': 1}

    assert species_counts == expected_species
    assert group_counts == expected_groups

# ---- Test Task 0.2.3: Extract file metadata ----
# Note: This test requires mocking the audio loading library (e.g., librosa)
# as we don't have real audio files with metadata.
#@pytest.mark.skip(reason="Requires mocking audio library like librosa")
def test_extract_file_metadata(temp_data_dir, mocker):
    # Mock soundfile.SoundFile to simulate reading file attributes
    mock_sf_file = MagicMock()
    mock_sf_file.frames = 336000 # Example frames
    mock_sf_file.samplerate = 32000 # Example samplerate
    # duration = frames / samplerate = 336000 / 32000 = 10.5
    mock_sf_context = MagicMock(__enter__=MagicMock(return_value=mock_sf_file))

    # Patch the soundfile.SoundFile context manager
    mocker.patch('soundfile.SoundFile', return_value=mock_sf_context)

    # We don't expect librosa to be called if soundfile succeeds
    mock_librosa_load = mocker.patch('librosa.load', side_effect=AssertionError("Librosa should not be called"))
    mock_librosa_duration = mocker.patch('librosa.get_duration', side_effect=AssertionError("Librosa should not be called"))

    file_path = str(temp_data_dir / "train_audio" / "bird1" / "file1.ogg")
    metadata = extract_file_metadata(file_path)

    assert metadata['file_path'] == file_path
    assert metadata['duration'] == 10.5
    assert metadata['sampling_rate'] == 32000
    assert metadata['format'] == '.ogg'
    assert metadata['error'] is None

#@pytest.mark.skip(reason="Requires mocking audio library like librosa")
def test_extract_file_metadata_error(temp_data_dir, mocker):
     # Mock librosa.load to raise an exception
    mocker.patch('librosa.load', side_effect=Exception("Failed to load"))
    mocker.patch('librosa.get_duration', side_effect=Exception("Failed to load")) # Mock duration too
    mocker.patch('soundfile.SoundFile', side_effect=Exception("Failed to load")) # Also mock soundfile

    # Use a file known to exist but cause error in mock
    file_path = str(temp_data_dir / "train_audio" / "bird1" / "file1.ogg")
    metadata = extract_file_metadata(file_path)

    assert metadata['file_path'] == file_path
    assert metadata['duration'] is None
    assert metadata['sampling_rate'] is None
    assert metadata['format'] == '.ogg'
    assert isinstance(metadata['error'], str)
    assert "Failed to load" in metadata['error']

def test_extract_file_metadata_non_audio(temp_data_dir):
    # Test with a non-audio file extension
    file_path = str(temp_data_dir / "train_audio" / "bird1" / "not_audio.txt")
    metadata = extract_file_metadata(file_path) # Should ideally handle gracefully

    assert metadata['file_path'] == file_path
    assert metadata['duration'] is None
    assert metadata['sampling_rate'] is None
    assert metadata['format'] == '.txt'
    # Depending on implementation, might have an error or just return None
    assert metadata['error'] is not None or (metadata['duration'] is None and metadata['sampling_rate'] is None)


# ---- Test Task 0.2.4: Create metadata DataFrame ----
def test_create_metadata_dataframe(temp_data_dir):
    # Sample metadata list (as if generated by extract_file_metadata)
    metadata_list = [
        {'file_path': 'path/bird1/f1.ogg', 'duration': 10.0, 'sampling_rate': 32000, 'format': '.ogg', 'error': None, 'primary_label': 'bird1'},
        {'file_path': 'path/bird2/f2.ogg', 'duration': 5.5, 'sampling_rate': 44100, 'format': '.ogg', 'error': None, 'primary_label': 'bird2'},
        {'file_path': 'path/bird1/f3.wav', 'duration': None, 'sampling_rate': None, 'format': '.wav', 'error': 'Load Error', 'primary_label': 'bird1'}
    ]
    taxonomy_path = temp_data_dir / "taxonomy.csv"
    taxonomy_df = pd.read_csv(taxonomy_path)

    metadata_df = create_metadata_dataframe(metadata_list, taxonomy_df)

    assert isinstance(metadata_df, pd.DataFrame)
    assert len(metadata_df) == 3
    assert 'file_path' in metadata_df.columns
    assert 'duration' in metadata_df.columns
    assert 'sampling_rate' in metadata_df.columns
    assert 'format' in metadata_df.columns
    assert 'error' in metadata_df.columns
    assert 'primary_label' in metadata_df.columns
    assert 'species_code' in metadata_df.columns
    assert 'class' in metadata_df.columns # Check if taxonomy info is merged

    # Check values
    assert metadata_df.loc[0, 'primary_label'] == 'bird1'
    assert metadata_df.loc[0, 'class'] == 'Aves'
    assert metadata_df.loc[1, 'sampling_rate'] == 44100
    assert pd.isna(metadata_df.loc[2, 'duration'])
    assert metadata_df.loc[2, 'error'] == 'Load Error'

# ---- Test Task 0.2.5: Generate summary statistics ----
def test_generate_summary_statistics(temp_data_dir):
     # Create a dummy metadata DataFrame
    taxonomy_path = temp_data_dir / "taxonomy.csv"
    taxonomy_df = pd.read_csv(taxonomy_path)
    data = {
        'file_path': ['p1', 'p2', 'p3', 'p4', 'p5'],
        'primary_label': ['bird1', 'bird1', 'bird2', 'frog1', 'insect1'],
        'duration': [10.0, 12.5, 5.0, 8.0, 20.0],
        'sampling_rate': [32000, 32000, 44100, 32000, 48000],
        'format': ['.ogg', '.ogg', '.wav', '.ogg', '.ogg'],
        'error': [None, None, None, None, 'Error X'] # One error example
    }
    metadata_df = pd.DataFrame(data)
    # Merge taxonomy info for group counts
    metadata_df = pd.merge(metadata_df, taxonomy_df[['primary_label', 'class']], on='primary_label', how='left')


    summary = generate_summary_statistics(metadata_df)

    assert isinstance(summary, dict)
    assert 'total_files' in summary
    assert 'total_species' in summary
    assert 'total_taxonomic_groups' in summary
    assert 'files_per_species' in summary
    assert 'files_per_group' in summary
    assert 'duration_stats' in summary
    assert 'sampling_rate_stats' in summary
    assert 'format_counts' in summary
    assert 'error_count' in summary

    # Check some values
    assert summary['total_files'] == 5
    assert summary['total_species'] == 4
    assert summary['total_taxonomic_groups'] == 3 # Aves, Amphibia, Insecta
    assert summary['files_per_species']['bird1'] == 2
    assert summary['files_per_group']['Aves'] == 3
    assert summary['duration_stats']['mean'] == pytest.approx(11.1) # Mean of non-NaN durations
    assert summary['sampling_rate_stats'][32000] == 3
    assert summary['format_counts']['.ogg'] == 4
    assert summary['error_count'] == 1
