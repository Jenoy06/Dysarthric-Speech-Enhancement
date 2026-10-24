# Dysarthria Detection and Severity Classification

Dysarthria is a motor speech disorder caused by neurological conditions that impair the muscle control needed for speech. This project explores automatic detection of dysarthric speech and classification of its severity using acoustic and deep learning features, evaluated on the TORGO and UASpeech datasets.

## Notebooks

| Notebook | Description |
|---|---|
| `mfcc_analysis.ipynb` | Dysarthria detection using MFCC features with classical classifiers |
| `pe_sfcc.ipynb` | Detection using Perceptually Enhanced Single Frequency Cepstral Coefficients (PE-SFCC) |
| `acoustic_analysis.ipynb` | Comparative acoustic feature analysis across dysarthric and control speakers |
| `wav2vec_analysis.ipynb` | Detection using wav2vec 2.0 embeddings on TORGO |
| `wav2vec_uaspeech.ipynb` | wav2vec 2.0 based analysis on the UASpeech dataset |
| `Submission/severity_classification.ipynb` | Severity level classification (mild / moderate / severe) |

## Datasets

Experiments use the [TORGO](http://www.cs.toronto.edu/~frank/TORGO/) and [UASpeech](https://www.isca-archive.org/interspeech_2008/kim08c_interspeech.html) corpora. Datasets are not included in this repository due to size and licensing.