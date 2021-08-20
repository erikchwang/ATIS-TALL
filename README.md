## ATIS-TALL

### setup
* en-ja-mbert: sh setup.sh en ja bert-base-multilingual-cased
* de-ja-mbert: sh setup.sh de ja bert-base-multilingual-cased
* en-de-mbert: sh setup.sh en de bert-base-multilingual-cased
* en-ja-xlmr: sh setup.sh en ja xlm-roberta-base
* de-ja-xlmr: sh setup.sh de ja xlm-roberta-base
* en-de-xlmr: sh setup.sh en de xlm-roberta-base

### launch
* pipeline: sh launch.sh
* preprocess: sh launch.sh 0
* pretrain: sh launch.sh 1
* finetune: sh launch.sh 2
* baseline: sh launch.sh 3
