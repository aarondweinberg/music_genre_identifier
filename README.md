# Identifying Music Genre  
**Aaron Weinberg, Emilie Wiesner, Dan Visscher**  
GitHub: [https://github.com/aarondweinberg/music_genre_identifier](https://github.com/aarondweinberg/music_genre_identifier)

## Introduction  
Music genre identification is useful not only for commercial applications like recommendation systems but also for deepening cultural engagement with music. Machine learning approaches have traditionally treated this as a “bucket sorting” problem, but music is experientially and culturally much more connected than discrete buckets. 

Although there has been extensive prior work on identifying the genre of recorded music, this work has been criticized for its narrow use of deep learning tools, dataset limitations, and lack of engagement with genre theory.[^1]

This project explores whether a neural network can be trained to recognize genre with greater contextual sensitivity, informed by genre theory and inter-genre relationships.

## Approach  
We used [Musicmap](https://musicmap.info/), a research-based web resource that categorizes ~250 modern Western music genres and defines inter-genre influences and “supergenre” clusters. From this, we built a weighted graph of genres, supergenres, and clusters of supergenres, and computed distances between them. Inspired by prior work on hierarchical loss functions,[^2] we also created a "soft" loss function that uses the weighted distance between genres, thereby enabling models to prioritize misclassifications that are “closer” in genre space.

## Data and Features  
Using Musicmap's curated playlists (~10 songs per genre), we split the songs in each genre into 70/20/10 train/validation/test sets. Following recommendations from the literature,[^3] we split each song into 15-second segments and augmented each segment with a version that was pitch-shifted by +1 semitone and a version with pink noise added at a scale of 0.005. We balanced the classes by randomly selecting 24 segments from each song. Then, we used librosa to engineer the following features for each segment:
- Chromagram
- Constant-Q chromagram
- Normalized chroma energy
- Mel-scaled spectrogram in decibels
- Mel-scaled spectrogram with per-channel energy normalization
- Graph of Mel-frequency cepstral coefficients
- Graphs of Harmonic-percussive source separation mean and median

## Model Creation and Training
For a baseline, we used progressive unfreezing to fine-tune a ResNet50 architecture with IMAGENET1K_V2 weights, using cross-entropy as the loss function.

We then created five architectures:
- A "ResNet18flex" architecture that accommodates smaller images by making adjustments to early stride/maxpool values
- A "ResNet18flex_dual" architecture that uses two features as inputs
- Custom CNN, GRU, and LSTM architectures suggested by the literature[^3]

We trained and evaluated performance of the ResNet18flex architectures using each type (and pair) of features, identifying the Mel-PCEN spectrograms as leading to the highest performance. We used the same architecture to tune the graph-weight hyperparameters.

We then tuned each model for a range of values of the hyperparameter $$\\beta$$, used in the Soft Loss function.

## Model Selection and Results  

To evaluate models, we introduced mean genre distance (based on the Musicmap graph), along with top-3 accuracy and supergenre accuracy. 

Our baseline model achieved:
- **Top-3 Accuracy:** 12.4%  
- **Top-3 Supergenre Accuracy:** 36.6%  
- **Mean Genre Distance:** 3.54  

Our best model—a dual-backbone ResNet18 using mel PCEN + MFCC features with a soft loss—achieved:  
- **Top-3 Accuracy:** 24.6%  
- **Top-3 Supergenre Accuracy:** 50.8%  
- **Mean Genre Distance:** 2.44  

Evaluating this model against the reserved test set data showed these metrics are stable.

## Key Insights  
- This project makes two significant contributions. The first is our creation of a graph to model the relationships and distance between genres and the second is our creation of a “soft loss” function to train a neural network based on these distances. This accommodates being “close” even if not completely correct.  
- The Musicmap playlist dataset is relatively small. With only 10 songs per genre, the variation within each genre is still too large to be well-represented. To avoid overfitting the training set, we need more data.  
- It is unclear whether audio is sufficient for determining genre using this more socially and historically informed notion of genre, particularly with the fine-grained approach taken here.  


## Files

### [musicmap_helpers.py](./musicmap_helpers.py)
A utility module used across most training and testing scripts. Includes functions to:
- Compute the mean and standard deviation of image folders
- Detect and configure GPU availability
- Implement soft labeling loss
- Compute top-k accuracy
- Evaluate validation batches, reporting genre/supergenre top-1/3/5 accuracy, mean/std distance, and confusion matrices
- Plot metrics across epochs
- Inspect image dimensions
- Create paired-feature datasets

### [Musicmap.html](./Musicmap.html)
Saved HTML of the [Musicmap.info](https://musicmap.info/) site, used to construct the genre graph and shortest-distance matrix for loss computation.

### [requirements.txt](./requirements.txt)
Dependency list to replicate the environment (e.g., in AzureML).

---

### [Deployment_local](./Deployment_local/)
Contains code and assets for deploying the model using Gradio:
- `best_model.pth`: Large file (not included)
- [class_names.txt](./Deployment_local/class_names.txt): Ordered genre names
- [gradio_app.py](./Deployment_local/gradio_app.py): Gradio interface that:
  - Splits audio into 15-second segments
  - Predicts genre/supergenre per segment
  - Returns top-3 predictions + distance-1 neighbors
- [model_resnet18flex_dual.py](./Deployment_local/model_resnet18flex_dual.py): Model architecture used
- [musicmap_graph_creator.py](./Deployment_local/musicmap_graph_creator.py): Constructs the genre graph
- [musicmap_helpers.py](./Deployment_local/musicmap_helpers.py): Utilities for evaluation
- [shortest_graph.pt](./Deployment_local/shortest_graph.pt): Precomputed shortest-path matrix

---

### [Feature Generation](./Feature%20Generation/)
Scripts for creating input features:
- [features_pipeline.ipynb](./Feature%20Generation/features_pipeline.ipynb): Notebook calling the feature extraction script
- [features_worker.py](./Feature%20Generation/features_worker.py): Splits audio into 15s segments, extracts Mel/Chroma/MFCC/etc. using `librosa`

---

### [Final Model Data and Outputs](./Final%20Model%20Data%20and%20Outputs/)
Contains evaluation outputs from final trained models:
- [Final_Models_Best_Metrics.csv](./Final%20Model%20Data%20and%20Outputs/Final_Models_Best_Metrics.csv): Training/validation metrics across all models
- [testing_ResNet_outputs_pcen_mfcc](./Final%20Model%20Data%20and%20Outputs/testing_ResNet_outputs_pcen_mfcc/):
  - Confusion matrices for genre/supergenre ([CSV](./Final%20Model%20Data%20and%20Outputs/testing_ResNet_outputs_pcen_mfcc/confusion_matrix_genre_TEST.csv), [heatmaps](./Final%20Model%20Data%20and%20Outputs/testing_ResNet_outputs_pcen_mfcc/supergenre_confusion_matrix_TEST.png))
  - Testing metrics ([CSV](./Final%20Model%20Data%20and%20Outputs/testing_ResNet_outputs_pcen_mfcc/testing_metrics.csv), [TXT](./Final%20Model%20Data%20and%20Outputs/testing_ResNet_outputs_pcen_mfcc/testing_metrics.txt))
- Output folders for other model variants (e.g. `trained_CNN_outputs_beta1`, `trained_ResNet_outputs_beta25`, etc.)

---

### [Graph Creation](./Graph%20Creation/)
Used to generate the graph structure for graph-aware loss:
- [musicmap_graph_creator.py](./Graph%20Creation/musicmap_graph_creator.py): Builds the graph and shortest-path matrix
- [musicmap_genres_withexplicitsupergenres.csv](./Graph%20Creation/musicmap_genres_withexplicitsupergenres.csv): Genre metadata including supergenres, clusters, and nodes

---

### [Models](./Models/)
Model architecture definitions:
- [model_CNN.py](./Models/model_CNN.py): CNN variant from literature
- [model_GRUl.py](./Models/model_GRUl.py): GRU variant
- [model_LSTM.py](./Models/model_LSTM.py): LSTM variant
- [model_resnet18flex.py](./Models/model_resnet18flex.py): Modified ResNet18 (lower stride, no early maxpool)
- [model_resnet18flex_dual.py](./Models/model_resnet18flex_dual.py): Dual-branch model for two input features

---

### [Training and Testing](./Training%20and%20Testing/)
Notebooks and scripts for model training/testing:
- [testing_ResNet18flex_dual_melpcen_mfcc.py](./Training%20and%20Testing/testing_ResNet18flex_dual_melpcen_mfcc.py)
- [training_CNN.py](./Training%20and%20Testing/training_CNN.py)
- [training_ResNet18flex_dual_melpcen_chromacq.py](./Training%20and%20Testing/training_ResNet18flex_dual_melpcen_chromacq.py)
- [training_ResNet18flex_dual_melpcen_mfcc.py](./Training%20and%20Testing/training_ResNet18flex_dual_melpcen_mfcc.py)
- [training_ResNet18flex.py](./Training%20and%20Testing/training_ResNet18flex.py)
- Notebooks:
  - [training_resnet50_pretrained_15secfeatures_crossentropy.ipynb](./Training%20and%20Testing/training_resnet50_pretrained_15secfeatures_crossentropy.ipynb)
  - [training_resnet50_pretrained_15secfeaturesaugmented_crossentropy.ipynb](./Training%20and%20Testing/training_resnet50_pretrained_15secfeaturesaugmented_crossentropy.ipynb)
  - [training_resnet50_pretrained_15secfeaturesaugmented_softlabeling.ipynb](./Training%20and%20Testing/training_resnet50_pretrained_15secfeaturesaugmented_softlabeling.ipynb)
  - [training_resnet50_pretrained_30secfeatures_crossentropy.ipynb](./Training%20and%20Testing/training_resnet50_pretrained_30secfeatures_crossentropy.ipynb)

---

### [Tuning](./Tuning/)
Code for hyperparameter and feature-space tuning:
- Architecture-specific beta sweeps:
  - [CNN](./Tuning/tuning_beta_CNN.py) / [GRU](./Tuning/tuning_beta_GRU.py) / [LSTM](./Tuning/tuning_beta_LSTM.py) / [ResNet18flex](./Tuning/tuning_beta_resnet18flex.py)
- [tuning_graphweights_resnet18flex_optuna.py](./Tuning/tuning_graphweights_resnet18flex_optuna.py): Graph weight tuning via Optuna
- [tuning_resnet18flex_cuda.ipynb](./Tuning/tuning_resnet18flex_cuda.ipynb) & [tuning_resnet18flex_mps.ipynb](./Tuning/tuning_resnet18flex_mps.ipynb): Hardware-accelerated training
- [feature_loop_resnet18flex.py](./Tuning/feature_loop_resnet18flex.py): Single-feature performance
- [feature_loop_resnet18flex_dual.py](./Tuning/feature_loop_resnet18flex_dual.py): Dual-feature performance

---

### [Utilities](./Utilities/)
Scripts for dataset prep, organization, and file management:
- [computing_means_and_sds.ipynb](./Utilities/computing_means_and_sds.ipynb): Computes image means/SDs
- [file_split_15sec_augmented.csv](./Utilities/file_split_15sec_augmented.csv): Train/val/test info and segment counts
- [genre_file_and_folder_renamer.py](./Utilities/genre_file_and_folder_renamer.py): Renames non-Musicmap genre labels
- [musicmap_data_splitter_featuresaligned.ipynb](./Utilities/musicmap_data_splitter_featuresaligned.ipynb): Splits data and prepares feature folders
- [Musicmap_Directory_Renaming.ipynb](./Utilities/Musicmap_Directory_Renaming.ipynb): Converts between Musicmap-style and human-readable folder names

[^1]: Green, O., Sturm, B., Born, G., & Wald-Fuhrmann, M. (2024). A critical survey of research in music genre recognition. *International Society for Music Information Retrieval*.
[^2]: Bertinetto, L., Mueller, R., Tertikas, K., Samangooei, S., & Lord, N. A. (2020). Making better mistakes: Leveraging class hierarchies with deep networks. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition* (pp. 12506-12515).
[^3]: Ba, T. C., Le, T. D. T., & Van, L. T. (2025). Music genre classification using deep neural networks and data augmentation. *Entertainment Computing, 53*, 100929.