# pca_forest_cleaner
RFI removal tool for pulsar archives. Written by Lars Künkel.

Uses principal component analysis and the isolation forest algorithm from scikit-learn to find RFI-polluted profiles.
The fraction of profiles that are treated as RFI is chosen by maximising the snr of pulse profile.
  
Usage:

  python pca_forest_cleaner.py -h
  
  python pca_forest_cleaner.py archive
