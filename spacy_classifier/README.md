## Spacy Classifier Setup

[Configuration Link](https://spacy.io/usage/training#quickstart)

`touch base_config.cfg`
- Paste configuration into `base_config.cfg`
- Open terminal and change to working directory `python -m spacy init fill-config ./base_config.cfg ./config.cfg`
- `mkdir output`
- Run our config file `python -m spacy train config.cfg --output output/`