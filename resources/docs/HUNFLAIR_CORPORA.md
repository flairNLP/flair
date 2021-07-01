# HunFlair - Data Sets
Here you can find an overview about biomedical NER data sets integrated in *HunFlair*.

__Content:__ [Overview](#overview) | [HUNER Data Sets](#huner-data-sets) | [BioBERT Evaluation Splits](#biobert-evaluation-splits)

## Overview
*HunFlair* integrates 31 biomedical named entity recognition (NER) data sets and provides 
them in an unified format to foster the development and evaluation of new NER models. All
data set implementations can be found in `flair.datasets.biomedical`.

| Corpus          | Data Set Class | Entity Types | Reference   | 
| ---             | --- | ---  | ---    |
| AnatEM | `ANAT_EM` | Anatomical entities | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3957068/), [Website](http://nactem.ac.uk/anatomytagger/#AnatEM)   |
| Arizona Disease | `AZDZ` | Disease | [Website](http://diego.asu.edu/index.php)   |
| BioCreative II GM | `BC2GM` | Gene | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2559986/)   |
| BioCreative V CDR task | `CDR` | Chemical, Disease  | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4860626/), [Website](https://github.com/JHnlp/BioCreative-V-CDR-Corpus)   |
| BioInfer | `BIO_INFER` |  Gene/Protein | [Paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-50)   |
| BioNLP'2013 Cancer Genetics (ST) | `BIONLP2013_CG` | Chemical, Disease, Gene/Protein, Species | [Paper](https://www.aclweb.org/anthology/W13-2008/)   |
| BioNLP'2013 Pathway Curation (ST)| `BIONLP2013_PC` | Chemical, Gene/Proteins  | [Paper](http://diego.asu.edu/index.php)   |
| BioSemantics| `BIOSEMANTICS` | Chemical, Disease | [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0107477), [Website](https://biosemantics.erasmusmc.nl/index.php/resources/chemical-patent-corpus)|
| CellFinder | `CELL_FINDER` | Cell line, Gene, Species  | [Paper](https://pdfs.semanticscholar.org/38e3/75aeeeb1937d03c3c80128a70d8e7a74441f.pdf)   |
| CEMP | `CEMP` | Chemical | [Website](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/cemp-detailed-task-description/)   |
| CHEBI | `CHEBI` | Chemical, Gene, Species  | [Paper](http://www.lrec-conf.org/proceedings/lrec2018/pdf/229.pdf)   |
| CHEMDNER | `CHEMDNER` | Chemical  | [Paper](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S2)   |
| CLL | `CLL` | Cell line  | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4708107/)   |
| DECA | `DECA` | Gene | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2828111/)   |
| FSU | `FSU` | Gene  | [Paper](https://www.aclweb.org/anthology/W10-1838/)   |
| GPRO | `GPRO` | Gene  | [Website](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/gpro-detailed-task-description/)   |
| CRAFT (v2.0) | `CRAFT` | Chemical, Gene, Species  | [Paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-13-161)  |
| CRAFT (v4.0.1) | `CRAFT_V4` | Chemical, Gene, Species  | [Website](https://github.com/UCDenver-ccp/CRAFT)   |
| GELLUS | `GELLUS` | Cell line  | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4708107/) |
| IEPA | `IEPA` | Gene  | [Paper](https://www.ncbi.nlm.nih.gov/pubmed/11928487) |
| JNLPBA | `JNLPBA` | Cell line, Gene  | [Paper](https://www.aclweb.org/anthology/W04-1213.pdf) |
| LINNEAUS | `LINNEAUS` | Species  | [Paper](https://www.ncbi.nlm.nih.gov/pubmed/20149233)   |
| LocText | `LOCTEXT` | Gene, Species  | [Paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2021-9)   |
| miRNA | `MIRNA` | Disease, Gene, Species  | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4602280/)   |
| NCBI Disease | `NCBI_DISEASE` | Disease  | [Paper](https://www.ncbi.nlm.nih.gov/pubmed/24393765)   |
| Osiris v1.2 | `OSIRIS` | Gene  | [Paper](https://www.ncbi.nlm.nih.gov/pubmed/18251998)   |
| Plant-Disease-Relations | `PDR` | Disease  | [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0221582), [Website](http://gcancer.org/pdr/)   |
| S800 | `S800` | Species  | [Paper](http://www.plosone.org/article/info:doi%2F10.1371%2Fjournal.pone.0065390)   |
| SCAI Chemicals | `SCAI_CHEMICALS` | Chemical  | [Paper](https://pub.uni-bielefeld.de/record/2603498)   |
| SCAI Disease | `SCAI_DISEASE` | Disease  | [Paper](https://pub.uni-bielefeld.de/record/2603398)   |
| Variome | `VARIOME` | Gene, Disease, Species  | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3676157/)   |

<sub>
Note: The table just gives an overview about the entity types of the individual corpora. 
Please refer to the original publications for annotation details. 
</sub>

## HUNER Data Sets
Next to the integration of the biomedical data sets, *HunFlair* provides the fixed splits used by 
[HUNER (Weber et al.)](https://academic.oup.com/bioinformatics/article/36/1/295/5523847) to 
improve comparability of evaluations

 | Entity Type  | Data Set Class   | Contained Data Sets | 
| ---           | ---              | ---                 |
| Cell Line    | `HUNER_CELL_LINE` | `HUNER_CELL_LINE_CELL_FINDER`, `HUNER_CELL_LINE_CLL`, `HUNER_CELL_LINE_GELLUS`, `HUNER_CELL_LINE_JNLPBA` |
| Chemical     | `HUNER_CHEMICAL`  | `HUNER_CHEMICAL_CDR`, `HUNER_CHEMICAL_CEMP`, `HUNER_CHEMICAL_CHEBI`, `HUNER_CHEMICAL_CHEMDNER`, `HUNER_CHEMICAL_CRAFT_V4`, `HUNER_CHEMICAL_SCAI` |
| Disease      | `HUNER_DISEASE`   | `HUNER_DISEASE_CDR`, `HUNER_DISEASE_MIRNA`, `HUNER_DISEASE_NCBI`, `HUNER_DISEASE_SCAI`, `HUNER_DISEASE_VARIOME` |
| Gene/Protein | `HUNER_GENE`      | `HUNER_GENE_BC2GM`, `HUNER_GENE_BIO_INFER`, `HUNER_GENE_CELL_FINDER`, `HUNER_GENE_CHEBI`, `HUNER_GENE_CRAFT_V4`, `HUNER_GENE_DECA`, `HUNER_GENE_FSU`, `HUNER_GENE_GPRO`, `HUNER_GENE_IEPA`, `HUNER_GENE_JNLPBA`, `HUNER_GENE_LOCTEXT`, `HUNER_GENE_MIRNA`, `HUNER_GENE_OSIRIS`, `HUNER_GENE_VARIOME` |
| Species      | `HUNER_SPECIES`   | `HUNER_SPECIES_CELL_FINDER`, `HUNER_SPECIES_CHEBI`, `HUNER_SPECIES_CRAFT_V4`, `HUNER_SPECIES_LINNEAUS`, `HUNER_SPECIES_LOCTEXT`, `HUNER_SPECIES_MIRNA`, `HUNER_SPECIES_S800`, `HUNER_SPECIES_VARIOME`|

## BioBERT evaluation splits
To ease comparison with BioBERT, *HunFlair* provides the splits used by 
[Lee et al.](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506):
`BIOBERT_GENE_BC4CHEMD`, `BIOBERT_GENE_BC2GM`, `BIOBERT_GENE_JNLPBA`, `BIOBERT_CHEMICAL_BC5CDR`,
`BIOBERT_DISEASE_BC5CDR`, `BIOBERT_DISEASE_NCBI`, `BIOBERT_SPECIES_LINNAEUS`, and `BIOBERT_SPECIES_S800`


Note: To download and use the BioBERT corpora you need to install the package _googledrivedownloader_, since the
files are hosted in Google Drive:
~~~
pip install googledrivedownloader
~~~ 




















