# Dataset description

Your challenge in this competition is to identify which species (birds, amphibians, mammals, insects) are calling in recordings made in El Silencio Natural Reserve, Colombia. This is an important task for scientists who monitor animal populations for conservation purposes. More accurate solutions could enable more comprehensive monitoring.

This competition uses a hidden test set. When your submitted notebook is scored, the actual test data will be made available to your notebook.

## Files

**train_audio/** The training data consists of short recordings of individual bird, amphibian, mammal and insects sounds generously uploaded by users of xeno-canto.org, iNaturalist and the Colombian Sound Archive (CSA) of the Humboldt Institute for Biological Resources Research in Colombia. These files have been resampled to 32 kHz where applicable to match the test set audio and converted to the ogg format. Filenames consist of [collection][file_id_in_collection].ogg. The training data should have nearly all relevant files; we expect there is no benefit to looking for more on xeno-canto.org or iNaturalist and appreciate your cooperation in limiting the burden on their servers. If you do, please make sure to adhere to the scraping rules of these data portals.

**test_soundscapes/** When you submit a notebook, the test_soundscapes directory will be populated with approximately 700 recordings to be used for scoring. They are 1 minute long and in ogg audio format, resampled to 32 kHz. The file names are randomized, but have the general form of soundscape_xxxxxx.ogg. It should take your submission notebook approximately five minutes to load all the test soundscapes. Not all species from the train data actually occur in the test data.

**train_soundscapes/** Unlabeled audio data from the same recording locations as the test soundscapes. Filenames consist of [site]_[date]_[local_time].ogg; although recorded at the same location, precise recording sites of unlabeled soundscapes do NOT overlap with recording sites of the hidden test data.

**train.csv** A wide range of metadata is provided for the training data. The most directly relevant fields are:

**primary_label:** A code for the species (eBird code for birds, iNaturalist taxon ID for non-birds). You can review detailed information about the species by appending codes to eBird and iNaturalis taxon URL, such as https://ebird.org/species/gretin1 for the Great Tinamou or https://www.inaturalist.org/taxa/24322 for the Red Snouted Tree Frog. Not all species have their own pages; some links might fail.
secondary_labels: List of species labels that have been marked by recordists to also occur in the recording. Can be incomplete.
latitude & longitude: Coordinates for where the recording was taken. Some bird species may have local call 'dialects,' so you may want to seek geographic diversity in your training data.
author: The user who provided the recording. Unknown if no name was provided.
filename: The name of the associated audio file.
rating: Values in 1..5 (1 - low quality, 5 - high quality) provided by users of Xeno-canto; 0 implies no rating is available; iNaturalist and the CSA do not provide quality ratings.
collection: Either XC, iNat or CSA, indicating which collection the recording was taken from. Filenames also reference the collection and the ID within that collection.

**sample_submission.csv** A valid sample submission.
row_id: A slug of soundscape_[soundscape_id]_[end_time] for the prediction; e.g., Segment 00:15-00:20 of 1-minute test soundscape soundscape_12345.ogg has row ID soundscape_12345_20.
[species_id]: There are 206 species ID columns. You will need to predict the probability of the presence of each species for each row.
taxonomy.csv - Data on the different species, including iNaturalist taxon ID and class name (Aves, Amphibia, Mammalia, Insecta).
recording_location.txt - Some high-level information on the recording location (El Silencio Natural Reserve).