# FoundPAD: Foundation Models Reloaded for Face Presentation Attack Detection
This repository contains the official implementation of the paper **"[FoundPAD: Foundation Models Reloaded for Face Presentation Attack Detection](https://arxiv.org/abs/2501.02892)"**, accepted at **WACV2025 AI4MFDD** (*2nd Workshop on Artificial Intelligence for Multimedia Forensics and Disinformation Detection*).

## Overview
FoundPAD leverages foundation models with LoRA adaptation to tackle the challenges of face presentation attack detection (PAD). It achieves state-of-the-art generalization across unseen domains and performs well under various data availability scenarios, including synthetic data.

<strong>Features:</strong>
- Foundation model adaptation with LoRA for PAD tasks.
- Generalization benchmarks on diverse datasets.
- Training pipelines for limited and synthetic data scenarios.
  
![Complete pipeline of FoundPAD](/img/pipeline.jpg)  
*Figure 1: Complete pipeline of FoundPAD. The proposed PAD model consists of an FM followed by a binary fully-connected classification layer. During training, the FM's feature space is adapted due to the training of the LoRA weights, while the classification layer is simultaneously trained to predict the PAD labels. It is better visualized in colour.*

![Integration of LoRA trainable weights](/img/mha_lora.jpg)  
*Figure 2: Integration of LoRA trainable weights (orange boxes) in a standard multi-head self-attention block, whose weights are kept frozen (blue boxes). In the proposed framework, FoundPAD, the LoRA adaptation is limited to the q and v matrices, leaving k and o unaltered. Better visualized in colour.*

## How to replicate

- Create a virtual environment by using **requirements.txt**
```
conda create -n env_name python=3.9
pip install -r requirements.txt
```
- Adjust config file in  **/src/config.py**
- Start training with  **./train.sh**

## Pre-trained Models and Training Logs

All pre-trained models and their respective training logs will be available soon.

## Key Results - ViT-B/16
|          |               |         |      | epoch = 40 |        |                |
|----------|---------------|---------|------|:----------:|:------:|----------------|
| FoundPAD |     Train     |         | Test |     AUC    |  HTER  |                |
| ViT-B/16 |       I       | Table 5 |   C  |   90,43%   | 17,00% |     Table 5    |
|          |               |         |   M  |   89,26%   | 18,57% |    Avg. AUC    |
|          |               |         |   O  |   93,74%   | 13,38% |     90,75%     |
|          |       C       |         |   I  |   91,50%   | 16,40% |    Avg. HTER   |
|          |               |         |   M  |   82,14%   | 24,52% |     16,81%     |
|          |               |         |   O  |   92,38%   | 15,14% | Std. Dev. HTER |
|          |       M       |         |   C  |   88,03%   | 20,00% |      5,03%     |
|          |               |         |   I  |   91,05%   | 17,10% |                |
|          |               |         |   O  |   88,63%   | 19,41% |     Table 2    |
|          |       O       |         |   C  |   97,11%   |  7,89% |    Avg. AUC    |
|          |               |         |   M  |   87,98%   | 23,33% |     95,52%     |
|          |               |         |   I  |   96,77%   |  8,95% |    Avg. HTER   |
|          |      M&I      | Table 4 |   C  |   93,97%   | 13,22% |     10,62%     |
|          |               |         |   O  |   96,69%   |  9,31% | Std. Dev. HTER |
|          |     O&C&M     | Table 3 |  CA  |   90,98%   | 15,65% |      7,29%     |
|          |               | Table 2 |   I  |   95,80%   | 10,45% |                |
|          |     O&C&I     |         |   M  |   89,88%   | 20,95% |     Table 6    |
|          |     O&M&I     |         |   C  |   98,08%   |  4,89% |    Avg. AUC    |
|          |     I&C&M     |         |   O  |   98,31%   |  6,19% |     78,39%     |
|          |  SynthASpoof  | Table 6 |   M  |   66,18%   | 47,14% |    Avg. HTER   |
|          |               |         |   C  |   83,03%   | 27,33% |     30,94%     |
|          |               |         |   I  |   90,79%   | 16,15% |                |
|          |               |         |   O  |   73,56%   | 33,12% |                |
|          | Total Average |         |      |   89,84%   | 17,66% |                |

|          |               |         |      | epoch = 40 |        |                |
|----------|---------------|---------|------|:----------:|:------:|----------------|
|  ViT-FE  |     Train     |         | Test |     AUC    |  HTER  |                |
| ViT-B/16 |       I       | Table 5 |   C  |   67,57%   | 38,56% |     Table 5    |
|          |               |         |   M  |   68,59%   | 35,71% |    Avg. AUC    |
|          |               |         |   O  |   64,66%   | 40,52% |     74,86%     |
|          |       C       |         |   I  |   73,46%   | 32,95% |    Avg. HTER   |
|          |               |         |   M  |   71,28%   | 35,24% |     31,89%     |
|          |               |         |   O  |   76,96%   | 30,63% | Std. Dev. HTER |
|          |       M       |         |   C  |   82,91%   | 25,33% |      6,29%     |
|          |               |         |   I  |   75,47%   | 30,45% |                |
|          |               |         |   O  |   77,66%   | 28,86% |     Table 2    |
|          |       O       |         |   C  |   91,07%   | 16,89% |    Avg. AUC    |
|          |               |         |   M  |   76,80%   | 32,86% |     80,32%     |
|          |               |         |   I  |   71,88%   | 34,65% |    Avg. HTER   |
|          |      M&I      | Table 4 |   C  |   79,94%   | 27,22% |     28,14%     |
|          |               |         |   O  |   72,70%   | 33,57% | Std. Dev. HTER |
|          |     O&C&M     | Table 3 |  CA  |   84,24%   | 23,66% |      7,32%     |
|          |               | Table 2 |   I  |   72,71%   | 36,10% |                |
|          |     O&C&I     |         |   M  |   77,50%   | 30,71% |     Table 6    |
|          |     O&M&I     |         |   C  |   90,33%   | 18,67% |    Avg. AUC    |
|          |     I&C&M     |         |   O  |   80,74%   | 27,07% |     72,21%     |
|          |  SynthASpoof  | Table 6 |   M  |   59,27%   | 47,14% |    Avg. HTER   |
|          |               |         |   C  |   78,81%   | 28,11% |     33,76%     |
|          |               |         |   I  |   87,08%   | 19,50% |                |
|          |               |         |   O  |   63,66%   | 40,28% |                |
|          | Total Average |         |      |   75,88%   | 31,07% |                |

|          |               |         |      | epoch = 40 |        |                |
|----------|---------------|---------|------|:----------:|:------:|----------------|
|  ViT-FS  |     Train     |         | Test |     AUC    |  HTER  |                |
| ViT-B/16 |       I       | Table 5 |   C  |   82,05%   | 24,33% |     Table 5    |
|          |               |         |   M  |   89,29%   | 22,38% |    Avg. AUC    |
|          |               |         |   O  |   89,10%   | 18,43% |     91,33%     |
|          |       C       |         |   I  |   83,51%   | 26,05% |    Avg. HTER   |
|          |               |         |   M  |   97,14%   |  8,33% |     15,88%     |
|          |               |         |   O  |   90,91%   | 17,79% | Std. Dev. HTER |
|          |       M       |         |   C  |   99,25%   |  4,00% |      7,07%     |
|          |               |         |   I  |   92,05%   | 15,05% |                |
|          |               |         |   O  |   98,10%   |  6,68% |     Table 2    |
|          |       O       |         |   C  |   95,33%   | 11,56% |    Avg. AUC    |
|          |               |         |   M  |   92,18%   | 15,48% |     95,99%     |
|          |               |         |   I  |   87,00%   | 20,45% |    Avg. HTER   |
|          |      M&I      | Table 4 |   C  |   92,97%   | 14,00% |     10,37%     |
|          |               |         |   O  |   97,88%   |  7,11% | Std. Dev. HTER |
|          |     O&C&M     | Table 3 |  CA  |   89,07%   | 16,01% |      3,87%     |
|          |               | Table 2 |   I  |   93,59%   | 14,90% |                |
|          |     O&C&I     |         |   M  |   96,09%   | 11,19% |     Table 6    |
|          |     O&M&I     |         |   C  |   95,67%   |  9,89% |    Avg. AUC    |
|          |     I&C&M     |         |   O  |   98,60%   |  5,52% |     63,91%     |
|          |  SynthASpoof  | Table 6 |   M  |   58,61%   | 50,24% |    Avg. HTER   |
|          |               |         |   C  |   59,46%   | 44,44% |     41,40%     |
|          |               |         |   I  |   81,48%   | 24,40% |                |
|          |               |         |   O  |   56,08%   | 46,53% |                |
|          | Total Average |         |      |   87,63%   | 18,90% |                |

## Key Results - ViT-L/14
|          |               |         |      | epoch = 40 |        |                |
|----------|---------------|---------|------|:----------:|:------:|----------------|
| FoundPAD |     Train     |         | Test |     AUC    |  HTER  |                |
| ViT-L/14 |       I       | Table 5 |   C  |   96,14%   | 10,22% |     Table 5    |
|          |               |         |   M  |   89,52%   | 19,29% |    Avg. AUC    |
|          |               |         |   O  |   90,79%   | 16,94% |     92,38%     |
|          |       C       |         |   I  |   93,12%   | 14,05% |    Avg. HTER   |
|          |               |         |   M  |   88,48%   | 21,43% |     15,49%     |
|          |               |         |   O  |   95,19%   | 11,00% | Std. Dev. HTER |
|          |       M       |         |   C  |   94,22%   | 12,00% |      5,07%     |
|          |               |         |   I  |   92,62%   | 14,55% |                |
|          |               |         |   O  |   87,37%   | 20,93% |     Table 2    |
|          |       O       |         |   C  |   97,64%   |  7,22% |    Avg. AUC    |
|          |               |         |   M  |   89,01%   | 23,81% |     96,60%     |
|          |               |         |   I  |   94,48%   | 14,40% |    Avg. HTER   |
|          |      M&I      | Table 4 |   C  |   99,22%   |  4,67% |      9,67%     |
|          |               |         |   O  |   95,58%   | 10,23% | Std. Dev. HTER |
|          |     O&C&M     | Table 3 |  CA  |   59,66%   | 42,99% |      5,17%     |
|          |               | Table 2 |   I  |   96,07%   |  9,90% |                |
|          |     O&C&I     |         |   M  |   93,18%   | 16,90% |     Table 6    |
|          |     O&M&I     |         |   C  |   98,72%   |  6,00% |    Avg. AUC    |
|          |     I&C&M     |         |   O  |   98,41%   |  5,87% |     85,01%     |
|          |  SynthASpoof  | Table 6 |   M  |   69,76%   | 45,71% |    Avg. HTER   |
|          |               |         |   C  |   96,03%   |  9,89% |     23,51%     |
|          |               |         |   I  |   98,58%   |  6,40% |                |
|          |               |         |   O  |   75,69%   | 32,05% |                |
|          | Total Average |         |      |   90,85%   | 16,37% |                |

|          |               |         |      | epoch = 40 |        |                |
|----------|---------------|---------|------|:----------:|:------:|----------------|
|  ViT-FE  |     Train     |         | Test |     AUC    |  HTER  |                |
| ViT-L/14 |       I       | Table 5 |   C  |   87,85%   | 19,78% |     Table 5    |
|          |               |         |   M  |   82,10%   | 26,90% |    Avg. AUC    |
|          |               |         |   O  |   74,33%   | 31,83% |     82,91%     |
|          |       C       |         |   I  |   81,37%   | 25,70% |    Avg. HTER   |
|          |               |         |   M  |   77,99%   | 30,24% |     24,43%     |
|          |               |         |   O  |   81,89%   | 25,06% | Std. Dev. HTER |
|          |       M       |         |   C  |   92,41%   | 15,44% |      6,21%     |
|          |               |         |   I  |   88,90%   | 19,00% |                |
|          |               |         |   O  |   79,36%   | 28,37% |     Table 2    |
|          |       O       |         |   C  |   93,24%   | 13,78% |    Avg. AUC    |
|          |               |         |   M  |   73,31%   | 32,62% |     88,52%     |
|          |               |         |   I  |   82,18%   | 24,40% |    Avg. HTER   |
|          |      M&I      | Table 4 |   C  |   94,57%   | 11,33% |     18,76%     |
|          |               |         |   O  |   81,31%   | 26,19% | Std. Dev. HTER |
|          |     O&C&M     | Table 3 |  CA  |   58,25%   | 43,86% |      6,51%     |
|          |               | Table 2 |   I  |   86,27%   | 22,05% |                |
|          |     O&C&I     |         |   M  |   86,87%   | 21,67% |     Table 6    |
|          |     O&M&I     |         |   C  |   96,10%   |  9,00% |    Avg. AUC    |
|          |     I&C&M     |         |   O  |   84,85%   | 22,32% |     78,41%     |
|          |  SynthASpoof  | Table 6 |   M  |   55,76%   | 52,62% |    Avg. HTER   |
|          |               |         |   C  |   92,82%   | 13,89% |     29,15%     |
|          |               |         |   I  |   87,94%   | 20,50% |                |
|          |               |         |   O  |   77,13%   | 29,58% |                |
|          | Total Average |         |      |   82,47%   | 24,61% |                |

|          |               |         |      | epoch = 40 |        |batch size = 412|
|----------|---------------|---------|------|:----------:|:------:|----------------|
|  ViT-FS  |     Train     |         | Test |     AUC    |  HTER  |                |
| ViT-L/14 |       I       | Table 5 |   C  |   83,28%   | 25,00% |     Table 5    |
|          |               |         |   M  |   87,37%   | 20,48% |    Avg. AUC    |
|          |               |         |   O  |   84,11%   | 23,62% |     87,42%     |
|          |       C       |         |   I  |   85,94%   | 22,60% |    Avg. HTER   |
|          |               |         |   M  |   97,74%   |  5,71% |     19,84%     |
|          |               |         |   O  |   68,58%   | 37,07% | Std. Dev. HTER |
|          |       M       |         |   C  |   96,39%   |  7,89% |      8,69%     |
|          |               |         |   I  |   94,67%   | 15,00% |                |
|          |               |         |   O  |   86,00%   | 22,93% |     Table 2    |
|          |       O       |         |   C  |   86,42%   | 20,33% |    Avg. AUC    |
|          |               |         |   M  |   95,72%   | 11,43% |     84,49%     |
|          |               |         |   I  |   82,87%   | 26,05% |    Avg. HTER   |
|          |      M&I      | Table 4 |   C  |   85,66%   | 25,22% |     23,04%     |
|          |               |         |   O  |   96,41%   |  9,07% | Std. Dev. HTER |
|          |     O&C&M     | Table 3 |  CA  |   52,29%   | 48,23% |     11,74%     |
|          |               | Table 2 |   I  |   87,29%   | 21,55% |                |
|          |     O&C&I     |         |   M  |   98,12%   |  8,10% |     Table 6    |
|          |     O&M&I     |         |   C  |   82,97%   | 26,11% |    Avg. AUC    |
|          |     I&C&M     |         |   O  |   69,57%   | 36,40% |     59,48%     |
|          |  SynthASpoof  | Table 6 |   M  |   55,94%   | 50,00% |    Avg. HTER   |
|          |               |         |   C  |   58,14%   | 47,11% |     45,19%     |
|          |               |         |   I  |   73,20%   | 33,60% |                |
|          |               |         |   O  |   50,63%   | 50,04% |                |
|          | Total Average |         |      |   80,84%   | 25,81% |                |

## Citation
```
@misc{ozgur2025foundpadfoundationmodelsreloaded,
      title={FoundPAD: Foundation Models Reloaded for Face Presentation Attack Detection}, 
      author={Guray Ozgur and Eduarda Caldeira and Tahar Chettaoui and Fadi Boutros and Raghavendra Ramachandra and Naser Damer},
      year={2025},
      eprint={2501.02892},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.02892}, 
}
```

## License
>This project is licensed under the terms of the **Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.  
Copyright (c) 2025 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt  
For more details, please take a look at the [LICENSE](./LICENSE) file.
