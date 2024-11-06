<div align="center">

<h1> PolyWER </h1>
This repository contains the implementation of the paper:

**PolyWER**: A Holistic Evaluation Framework for Code-Switched Speech Recognition

<a href=''> <a href=''><img src='https://img.shields.io/badge/Paper-ACL Anthology-red'></a> &nbsp;  <a href='https://huggingface.co/datasets/sqrk/mixat-tri'><img src='https://img.shields.io/badge/Dataset-Mixat-green'></a> &nbsp; [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]


<div>
    <a href='https://www.linkedin.com/in/karima-kadaoui-960923b7/' target='_blank'>Karima Kadaoui</a>&emsp;
    <a href='https://www.linkedin.com/in/maryam-al-ali-76b978231' target='_blank'>Maryam Al Ali</a>&emsp;
    <a href='https://www.linkedin.com/in/toyinhawau/' target='_blank'>Hawau Olamide Toyin</a>&emsp;
    <a href='https://www.linkedin.com/in/ibrahim-mohammed13' target='_blank'>Ibrahim Ali Mohammed</a>&emsp;
    <a href='https://linkedin.com/in/hanan-aldarmaki/' target='_blank'>Hanan Aldarmaki </a>&emsp;
</div>
<br>
<p align="center" float="center">
  <img src="img/MBZUAI-logo.png" height="40" />
</p>

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
We used the [Mixat](https://github.com/mbzuai-nlp/mixat) dataset for our experiments. The original dataset only contains the transcriptions with the English code-switching in latin characters. We augment these transcriptions with two additional dimensions: transliterations and translations. These can be found on [HuggingFace](https://huggingface.co/datasets/sqrk/mixat-tri).

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


## License
This data is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]


## Acknowledgements
If you use PolyWER, please cite the following paper:

``` 
@inproceedings{,
  
}

```

If you use the Mixat dataset (audio and\or text), please also cite the following paper:

``` 
@inproceedings{al-ali-aldarmaki-2024-mixat,
    title = "Mixat: A Data Set of Bilingual Emirati-{E}nglish Speech",
    author = "Al Ali, Maryam Khalifa  and
      Aldarmaki, Hanan",
    booktitle = "Proceedings of the 3rd Annual Meeting of the Special Interest Group on Under-resourced Languages @ LREC-COLING 2024",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.sigul-1.26",
    pages = "222--226"
}

```

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
