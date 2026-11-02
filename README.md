# Multi-Modal Deep Neural Network for Multi-Class Liver Cancer Diagnosis

This repository contains the source code and supporting scripts associated with the following publication:

**Khan RA, Fu M, Burbridge B, Luo Y, Wu FX. A multi-modal deep neural network for multi-class liver cancer diagnosis. _Neural Networks_. 2023;165:553–561.**  
https://doi.org/10.1016/j.neunet.2023.06.013

## Overview

The code provided in this repository was developed as part of the research presented in the publication above.

The study proposes a **multi-modal deep neural network for multi-class liver cancer diagnosis**, integrating information from multiple data modalities within a deep-learning framework.

This repository is made available to support:

- research reproducibility;
- academic and educational use;
- comparison with related machine-learning and deep-learning methods;
- further methodological development; and
- extension of the proposed multi-modal framework.

Users should refer to the original publication for complete details regarding the methodology, model architecture, datasets, preprocessing procedures, experimental design, and evaluation.

---

## Repository Structure

The primary scripts in this repository are:

| File | Description |
|---|---|
| `CAD*` | **Main script** used to run the proposed algorithm and associated experimental workflow. |
| `data_cad*` | Supporting/dependent script containing components required by the main algorithm, including the **neural network architecture, data preprocessing pipeline, and related functions**. |

In general, the workflow can be represented as:

```text
Input Data
    |
    v
Data Preparation / Preprocessing
(data_cad)
    |
    v
Neural Network Architecture
(data_cad)
    |
    v
Main Experimental Pipeline
(CAD)
    |
    v
Model Training / Evaluation
    |
    v
Multi-Class Liver Cancer Diagnosis
```

The `CAD` script depends on functions and components implemented in `data_cad`. Users should therefore ensure that both files are available in the appropriate project directory before running the algorithm.

---

## Usage

The repository contains **research code** corresponding to the methodology described in the associated publication.

Users may need to modify:

- dataset paths;
- directory structures;
- input data locations;
- software dependencies;
- training parameters; and
- environment-specific configurations

before executing the scripts.

The code should be interpreted and used together with the methodology described in the original article.

---

## Citation

If you use, modify, adapt, reproduce, extend, benchmark, or otherwise incorporate this code or the methodology implemented in this repository in any **publication, preprint, thesis, dissertation, conference paper, report, or other scholarly work**, you must appropriately acknowledge the original work and cite:

> Khan RA, Fu M, Burbridge B, Luo Y, Wu FX. A multi-modal deep neural network for multi-class liver cancer diagnosis. *Neural Networks*. 2023;165:553–561.  
> https://doi.org/10.1016/j.neunet.2023.06.013

### BibTeX

```bibtex
@article{khan2023multimodal,
  title={A multi-modal deep neural network for multi-class liver cancer diagnosis},
  author={Khan, RA and Fu, M and Burbridge, B and Luo, Y and Wu, FX},
  journal={Neural Networks},
  volume={165},
  pages={553--561},
  year={2023},
  doi={10.1016/j.neunet.2023.06.013}
}
```

Citation of the associated publication provides appropriate attribution to the original methodology and supports research reproducibility.

---

## Terms of Use

**All rights reserved.**

The source code in this repository is provided primarily for **academic, educational, and non-commercial research purposes**.

Use of the software is subject to the conditions described in the accompanying `LICENSE` file, where applicable.

### Academic and Non-Commercial Use

The code may be used for academic, educational, and non-commercial research purposes subject to the applicable licensing conditions.

Any scholarly work resulting from the use, modification, adaptation, reproduction, or extension of this code should provide appropriate attribution and cite the associated publication.

### Commercial Use

**Commercial use of this code, or substantial portions of it, is not permitted without prior written authorization from the applicable authors and/or copyright holders.**

Commercial use may include incorporation of the code into:

- commercial software;
- proprietary products;
- paid services;
- commercial clinical or diagnostic systems;
- commercial research and development activities; or
- other applications intended primarily for commercial advantage.

Organizations or individuals interested in commercial use should contact the authors/rightsholders to discuss appropriate authorization or licensing.

Public availability of this repository should not be interpreted as granting unrestricted commercial rights.

---

## Modification and Redistribution

Any permitted modification or redistribution of the source code must comply with the applicable license.

Where redistribution is permitted, the following should be preserved:

- attribution to the original authors;
- reference to the original publication;
- applicable copyright notices; and
- applicable license information.

Modified versions should be clearly identified as modifications and should not be represented as the original implementation produced, tested, or validated by the authors.

---

## Data Availability

The datasets used in the original study are **not automatically included or licensed through this repository** unless explicitly stated otherwise.

Users are responsible for independently obtaining appropriate datasets and complying with all applicable:

- dataset licenses;
- data-use agreements;
- institutional requirements;
- ethical approvals;
- privacy requirements; and
- other applicable conditions of data access.

Please refer to the original publication for details regarding the data and experimental methodology used in the study.

---

## Research and Clinical Disclaimer

This repository contains **research software** developed for scientific investigation.

The code is not provided as a certified medical device or as a clinically validated diagnostic system.

It should not be used as the sole basis for:

- clinical diagnosis;
- treatment decisions;
- patient management; or
- other medical decision-making.

Users are responsible for independently validating the software for their intended application and ensuring compliance with applicable regulatory, ethical, institutional, privacy, and legal requirements.

---

## No Warranty

The software is provided **"AS IS"**, without warranty of any kind, express or implied.

The authors make no guarantees regarding:

- accuracy;
- reliability;
- reproducibility;
- performance;
- clinical suitability; or
- fitness for any particular purpose.

The authors shall not be responsible for any damages, losses, clinical outcomes, or other consequences arising from the use or misuse of the software to the extent permitted by applicable law.

---

## Third-Party Software and Data

This repository may depend on external libraries, frameworks, datasets, or other third-party resources.

Such materials remain subject to their respective licenses, copyrights, terms of use, and data-use requirements.

The availability of this repository does not grant any rights to third-party datasets, medical images, libraries, software, or other protected materials used in the original study.

---

## Reproducibility

The code is released to improve transparency and reproducibility of the research presented in the associated publication.

Differences in:

- software versions;
- hardware;
- datasets;
- preprocessing procedures;
- random initialization;
- library implementations; or
- computing environments

may result in differences from the numerical results reported in the original publication.

Users should consult the published article for complete information regarding the experimental methodology.

---

## License

Copyright © Rayyan A. Khan, M. Fu, B. Burbridge, Y. Luo, F.-X. Wu. All rights reserved.

This repository is provided subject to the terms specified in the accompanying `LICENSE` file.

**Academic and non-commercial research use may be permitted under the applicable license, while commercial use requires prior authorization where specified.**

Users should review the complete license before using, modifying, redistributing, or incorporating the source code into another project.
