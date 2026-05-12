# Project to test super line graph architectures

This is only a partial implementation of my project.

Main structure:
```
to_present/
├── configs/          
│   ├── data/                            # yml files for datasets
│   │   ├── enzymes.yaml
│   │   ├── mutag.yaml
│   │   ├── nci1.yaml
│   │   └── zinc.yaml
│   ├── model/
│   │   ├── gcn.yaml
│   │   ├── mod_slg2.yaml
│   │   ├── mod_slg2_v2.yaml
│   │   ├── slg_advanced.yaml
│   │   ├── slg_naive.yaml
│   │   └── slg_v1.yaml
│   └── train.yaml                      # main configurations
├── src/
│   ├── data/                           # datasets handling
│   │   ├── stat_data/
│   │   ├── __init__.py
│   │   ├── data_loaders.py
│   │   ├── prep.py
│   │   ├── preprocessing.py
│   │   ├── slg2.cpp
│   │   └── transformation.py
│   ├── models/                         # many models has been implemented
│   │   ├── __init__.py
│   │   ├── classifier_head.py
│   │   ├── factory.py
│   │   ├── gcn.py
│   │   ├── mod_slg2.py
│   │   ├── mod_slg2_v2.py
│   │   ├── slg_advance.py
│   │   ├── slg_naive.py
│   │   └── slg_v1.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── graphgym_mapper.py          # to finish
│   │   ├── io.py
│   │   ├── metrics.py
│   │   └── viz.py
│   ├── __init__.py
│   ├── engine.py
│   └── train.py                        # main  
├── .gitignore
├── Readme.md
├── requirements.txt
└── setup.py
```

- The implementation of graphGym has to be completed.

