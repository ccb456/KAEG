# KEGA

Code for paper Knowledge-augmented and Evidence-guided document-level relation extraction with Axial Attention.

> Document-level relation extraction (DocRE) aims to identify semantic relationships between entities spanning multiple sentences, requiring holistic reasoning across pattern recognition, coreference resolution, common-sense inference, and logical deduction. Existing methods often prioritize isolated reasoning types, lacking integrated mechanisms for knowledge-guided and evidence-driven inference in long documents. To overcome these challenges, we propose a novel reasoning framework, Knowledge-augmented and Evidence-guided document-level relation extraction with axial attention (KEGA). Our approach reorganizes reasoning into three synergistic modules: (1) the document graph representation enhancement module, which constructs a multi-level heterogeneous graph and incorporates coreference resolution to enrich entity representations, supporting pattern recognition and coreference reasoning; (2) the knowledge-augmented module, which integrates an external knowledge base (e.g., Wikidata) to augment the document graph and employs a confidence-score-based filtering mechanism to mitigate knowledge noise, enhancing common-sense reasoning; and (3) the evidence-guided logical reasoning module, which leverages axial attention to capture long-range dependencies and applies evidence distribution supervision to optimize multi-step inference, bolstering logical reasoning. This unified architecture innovatively fuses multiple reasoning paradigms, enabling robust handling of cross-sentence complexities and implicit relations. Extensive experiments demonstrate that KEGA achieves state-of-the-art performance on Re-DocRED and DocRED datasets.



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

