<a name="readme-top"></a>
# Vocab-Guru: Vocabulary Expansion Tool for Language Learners
<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="static/vocab-guru.png" alt="Logo" width="240" height="240">
</div>
<br/>

Welcome to Vocab-Guru, A Vocabulary Expansion Tool for Language Learners. This tool aims to provide learners of the English language an opportuning to enhance their vocabulary by suggesting multiple improved versions of an input sentence by leveraging the capabilities of Modern LLMs like Llama2 and ChatGPT4. The tool also provides suggestions for better equivalents for given words, helping in vocabulary expansion.
The codes were run on a Multi-Instance GPU Partition of a NVIDIA A100.80GB GPU (40GB MIG Slice). The UI interface is powered by Flask in Python.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#About">About</a>
    </li>
    <li>
      <a href="#Getting-Started">Getting Started</a>
      <ul>
        <li><a href="#Prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li>
      <a href="#Fine-Tuning">Fine-Tuning</a>
      <ul>
        <li><a href="#Fine-Tuning-Llama2">Fine-Tuning Llama2</a></li>
        <li><a href="#Saving-the-Tokenizer">Saving the Tokenizer</a></li>
      </ul>
    </li>
    <li>
      <a href="#Using-Vocab-Guru">Using Vocab Guru</a>
    </li>
  </ol>
</details>


## About

Vocab-Guru, A Vocabulary Expansion Tool for Language Learners

## Getting Started

Clone the repository to your local machine:

```sh
git clone https://github.com/Vocab-Guru
cd Vocab-Guru
```

### Prerequisites

- Python installed (preferably the latest available version).

Install necessary libraries using the following command:

```sh
pip install -r requirements.txt
```

Upon successfully installing packages, install Torch using the following command:

```sh
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

You're all set!

## Fine-tuning

### Fine-tuning Llama2
Run the following code to Fine-tune the model:

```sh
python fine_tuning_llama_2.py
```

### Saving the Tokenizer

Once the model is trained, run this code to ensure that the Tokenizer is saved to avoid any issues while loading the model for prompting.

```sh
python save_tokenizer.py
```

## Using Vocab-Guru

```sh
python app.py
```
follow the link generated to access the Web UI interface and enjoy !!

<p align="right">(<a href="#readme-top">back to top</a>)</p>
