<div align="center">

<h1> PolyWER </h1>
This repository contains the implementation of the paper:

**PolyWER**: A Holistic Evaluation Framework for Code-Switched Speech Recognition

<a href=''> <a href=''><img src='https://img.shields.io/badge/paper-Paper-red'></a> &nbsp;  <a href='https://huggingface.co/datasets/sqrk/mixat-tri'><img src='https://img.shields.io/badge/data-Dataset-green'></a> &nbsp; 


<div>
    <a href='https://www.linkedin.com/in/karima-kadaoui-960923b7/' target='_blank'>Karima Kadaoui</a>&emsp;
    <a href='https://www.linkedin.com/in/maryam-al-ali-76b978231' target='_blank'>Maryam Al Ali</a>&emsp;
    <a href='https://www.linkedin.com/in/toyinhawau/' target='_blank'>Hawau Olamide Toyin</a>&emsp;
    <a href='https://www.linkedin.com/in/ibrahim-mohammed13' target='_blank'>Ibrahim Ali Mohammed</a>&emsp;
    <a href='https://linkedin.com/in/hanan-aldarmaki/' target='_blank'>Hanan Aldarmaki </a>&emsp;
</div>
<br>
<p align="left" float="center">
  <img src="img/MBZUAI-logo.png" height="40" />
</p>

<br>
<i><strong><a href='' target='_blank'>EMNLP 2024</a></strong></i>
<br>
</div>

## PolyWER 
Code-switching in speech can be correctly transcribed in various forms, including different ways of transliteration of the embedded language. Traditional metrics such as Word Error Rate (WER) are too strict to address this challenge. We introduce PolyWER, a framework for evaluating ASR systems to handle language-mixing. PolyWER accepts transcriptions of code-mixed segments in different forms, including transliterations and translations.

## Environment & Installation

Python version: 3.10+

```bash
git clone https://github.com/mbzuai-nlp/PolyWER.git
cd PolyWER
conda create -n polywer python=3.10
conda activate polywer
pip install -r requirements.txt
python toy_example.py
```

The toy example includes two sentences with their 3 transcription dimensions and outputs the different metrics we've used in our evaluation (PolyWER, WER, CER, BLEU, BERTScore).
To run [multiRefWer](https://github.com/qcri/multiRefWER), please do the following:

```bash
git clone https://github.com/qcri/multiRefWER 
cd multiRefWER
mrwer.py -e <polywer_path>/ref_og <polywer_path>/ref_lit <polywer_path>/ref_lat <polywer_path>/hyp 
```

Please note that we had to modify the mrwer code slightly to be able to run it (adding parentheses to the print statements and commenting out sys.reload)


## Dataset
The Mixat dataset with the additional transcriptions can be found on [HuggingFace](https://huggingface.co/datasets/sqrk/mixat-tri).

```python
>>> from datasets import load_dataset
>>> mixat = load_dataset("sqrk/mixat-tri")
>>> mixat
DatasetDict({
    train: Dataset({
        features: ['audio', 'transcript', 'transliteration', 'translation', 'language', 'duration_ms'],
        num_rows: 3727
    })
    test: Dataset({
        features: ['audio', 'transcript', 'transliteration', 'translation', 'language', 'duration_ms'],
        num_rows: 1585
    })
})

```


# Acknowledgements
If you use any PolyWER, please cite 

``` 
@inproceedings{,
  
}
```
