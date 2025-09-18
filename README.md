# KEGA

Code for paper **Knowledge-augmented and Evidence-guided document-level relation extraction with Axial Attention.**
> Document-level relation extraction (DocRE) seeks to identify semantic relationships between entities across multiple sentences, requiring holistic reasoning such as pattern recognition, coreference resolution, and common-sense or logical inference. However, existing methods often address these reasoning skills in isolation, lacking a unified framework for comprehensive inference. To address this limitation, we propose Knowledge-augmented and Evidence-Guided document-level relation extraction with Axial Attention (KEGA). KEGA integrates three complementary modules: (i) a document graph representation enhancement module, which constructs a multi-level heterogeneous graph and incorporates coreference resolution to enrich entity representations; (ii) a knowledge-augmented module that introduces external knowledge with confidence filtering to reduce noise; and (iii) an evidence-guided
logical reasoning module that employs axial attention and evidence supervision to refine multi-step inference. This unified architecture fuses multiple reasoning paradigms, enabling robust handling of cross-sentence dependencies and implicit relations. Extensive experiments demonstrate that KEGA achieves state-of-the-art performance on Re-DocRED and DocRED datasets.

## Requirements

Packages listed below are required.

- python==3.10
- CUDA==12.1
- PyTorch==2.1.0
- spacy==3.7.5
- tqdm==4.65.0
- requests==2.32.3
- beautifulsoup4==4.12.3
- wikidata==0.8.1
- dgl-cu121
- transformers==4.45.2
- rich==13.7.1
- omegaconf==2.3.0
- hydra-core==1.3.2
- axial-attention==0.6.1
- opt_einsum==3.4.0



## Datasets

Our experiments include the [DocRED](https://github.com/thunlp/DocRED) and [Re-DocRED](https://github.com/tonytan48/Re-DocRED) datasets. The expected file structure is as follows:

```
KEGA
├── dataset
│   ├── DocRED
│   |   ├── kg
│   |   │   ├── dev_graph.json
│   |   │   ├── test_graph.json
│   |   │   └── train_annotated_graph.json
│   │   ├── meta
│   │   ├── ref
│   │   ├── dev.json
│   │   ├── dev_coref.json
│   │   ├── test.json
│   │   ├── test_coref.json
│   │   ├── train_annotated.json
│   │   ├── train_annotated_coref.json
│   │   └── train_distant.json
│   └── Re-DocRED
│       ├── kg
│       │   ├── dev_revised_graph.json
│       │   ├── test_revised_graph.json
│       │   └── train_revised_graph.json
│       ├── meta
│       ├── ref
│       ├── dev_revised.json
│       ├── dev_revised_coref.json
│       ├── test_revised.json
│       ├── test_revised_coref.json
│       ├── train_distant.json
│       ├── train_revised.json
│       ├── train_revised_coref.json
│   ├── gen_coref.py
│   ├── gen_graph.py
│   ├── README.md
│   └── requirements.txt
```



## Training

If your dataset folder does not contain the kg and xxx-coref.json files, we recommend that you first run the gen_coref. py and gen_graph. py files under the dataset file, which generate corresponding reference files and knowledge graphs based on the dataset file. After obtaining these files, start executing the training code.

### DocRED

The corresponding parameter configuration for the model can be found in configs/train_docred.yaml. After configuring the model parameters, you only need to execute the following command:

```
python train_docred.py
```

### Re-DocRED

The corresponding parameter configuration for the model can be found in configs/train.yaml. After configuring the model parameters, you only need to execute the following command:

```
python train.py
```



## Evaluation

You need to adjust the corresponding parameter configuration file configs/xxx. yaml for the dataset, such as the model path. After completing the adjustment, you can execute the following command:

```
# Re-DocREd
python train.py

# DocRED
python train_docred.py
```

