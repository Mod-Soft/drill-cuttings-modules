This repository contains the official implementation of the core components proposed in our paper: Mitigating Misclassification of Confusable Drill Cuttings Categories via Dynamic Receptive Fields, Wavelet Frequency Decomposition, and Prior-Guided Fusion.

Files Included:

    DRFF.py: Dynamic Receptive Field Fusion module.

    FAW.py: Frequency Attentive Wavelet module.

    PGF.py: Prior-Guided Fusion module.

Usage Note:
To utilize these modules, it is necessary to implement a custom "module manager" to control the injection mechanism. Users can custom-build this manager, tailoring the integration strategy according to the specific architecture of the backbone network being modified.
