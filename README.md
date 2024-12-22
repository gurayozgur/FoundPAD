# FoundPAD: Foundation Models Reloaded for Face Presentation Attack Detection
This repository contains the official implementation of the paper **"[FoundPAD: Foundation Models Reloaded for Face Presentation Attack Detection](link-to-paper),"** submitted to **WACV2025 AI4MFDD** (*2nd Workshop on Artificial Intelligence for Multimedia Forensics and Disinformation Detection*).

## Overview
FoundPAD leverages foundation models with LoRA adaptation to tackle the challenges of face presentation attack detection (PAD). It achieves state-of-the-art generalization across unseen domains and performs well under various data availability scenarios, including synthetic data.

## Features
- Foundation model adaptation with LoRA for PAD tasks.
- Generalization benchmarks on diverse datasets.
- Training pipelines for limited and synthetic data scenarios.
  
![Complete pipeline of FoundPAD](img/pipeline.pdf)  
*Complete pipeline of FoundPAD. The proposed PAD model consists of an FM followed by a binary fully-connected classification layer. During training, the FM's feature space is adapted due to the training of the LoRA weights, while the classification layer is simultaneously trained to predict the PAD labels. It is better visualized in colour.*

![Integration of LoRA trainable weights](img/mha_lora.pdf)  
*Integration of LoRA trainable weights (orange boxes) in a standard multi-head self-attention block, whose weights are kept frozen (blue boxes). In the proposed framework, FoundPAD, the LoRA adaptation is limited to the $q$ and $v$ matrices, leaving $k$ and $o$ unaltered. Better visualized in colour.*


## Key Results - ViT-B/16



## Key Results - ViT-L/14




## Citation

@inproceedings{foundpad2025,
  title={FoundPAD: Foundation Models Reloaded for Face Presentation Attack Detection},
  author={
    Guray Ozgur, Eduarda Caldeira, Tahar Chettaoui, Fadi Boutros, Raghavendra Ramachandra, Naser Damer
  },
  booktitle={WACV AI4MFDD Workshop},
  year={2025},
  institution={Fraunhofer IGD, Fraunhofer IGD and TU Darmstadt, Norwegian University of Science and Technology},
  address={Germany, Norway},
  email={guray.ozgur@igd.fraunhofer.de, maria.eduarda.loureiro.caldeira@igd.fraunhofer.de, tahar.chettaoui@igd.fraunhofer.de, fadi.boutros@igd.fraunhofer.de, raghavendra.ramachandra@ntnu.no, naser.damer@igd.fraunhofer.de}
}

## License

This project is licensed under the terms of the **Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.  
Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt  
For more details, please take a look at the [LICENSE](./LICENSE) file.
