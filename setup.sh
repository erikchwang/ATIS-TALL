ROOT=$(dirname $(realpath ${0}))
mkdir ${ROOT}/anaconda ${ROOT}/source ${ROOT}/target ${ROOT}/task ${ROOT}/transformers ${ROOT}/data ${ROOT}/checkpoint
mkdir ${ROOT}/source/corpus ${ROOT}/source/stanza ${ROOT}/target/corpus ${ROOT}/target/stanza ${ROOT}/task/train ${ROOT}/task/develop ${ROOT}/task/test

wget -P ${ROOT} https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
wget -P ${ROOT} https://dumps.wikimedia.org/${1}wiki/latest/${1}wiki-latest-pages-articles-multistream.xml.bz2
wget -P ${ROOT} https://dumps.wikimedia.org/${2}wiki/latest/${2}wiki-latest-pages-articles-multistream.xml.bz2
wget -P ${ROOT} https://www.dropbox.com/s/d014qpiqxvsodye/multiatis.zip

sh ${ROOT}/Miniconda3-latest-Linux-x86_64.sh -b -f -p ${ROOT}/anaconda
source ${ROOT}/anaconda/bin/activate ${ROOT}/anaconda
conda install -y python=3.6
pip install -f https://download.pytorch.org/whl/torch_stable.html torch==1.6.0+cu101
pip install conda-pack==0.5.0 conlleval==0.2 nltk==3.5 stanza==1.1.1 transformers==3.1.0 wikiextractor==0.1

python -m wikiextractor.WikiExtractor --compress --json --output ${ROOT}/source/corpus ${ROOT}/${1}wiki-latest-pages-articles-multistream.xml.bz2
python -m wikiextractor.WikiExtractor --compress --json --output ${ROOT}/target/corpus ${ROOT}/${2}wiki-latest-pages-articles-multistream.xml.bz2
unzip ${ROOT}/multiatis.zip *${1}.tsv -d ${ROOT}/task/train
unzip ${ROOT}/multiatis.zip *${2}.tsv -d ${ROOT}/task/develop
unzip ${ROOT}/multiatis.zip *${2}.tsv -d ${ROOT}/task/test

python -c "
import stanza, transformers;
stanza.download('${1}', '${ROOT}/source/stanza', processors='tokenize');
stanza.download('${2}', '${ROOT}/target/stanza', processors='tokenize');
transformers.AutoConfig.from_pretrained('${3}').save_pretrained('${ROOT}/transformers');
transformers.AutoTokenizer.from_pretrained('${3}').save_pretrained('${ROOT}/transformers');
transformers.AutoModel.from_pretrained('${3}').save_pretrained('${ROOT}/transformers')
"

conda pack -o ${ROOT}/anaconda.tar.gz
conda deactivate
rm -rf ${ROOT}/anaconda
mkdir ${ROOT}/anaconda
tar -xzvf ${ROOT}/anaconda.tar.gz -C ${ROOT}/anaconda
rm ${ROOT}/anaconda.tar.gz

rm ${ROOT}/Miniconda3-latest-Linux-x86_64.sh
rm ${ROOT}/${1}wiki-latest-pages-articles-multistream.xml.bz2
rm ${ROOT}/${2}wiki-latest-pages-articles-multistream.xml.bz2
rm ${ROOT}/multiatis.zip
