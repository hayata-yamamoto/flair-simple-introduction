# flair-simple-introduction

## What's this? 

The purpose of this repository is to show flair usage, which I wrote on my article. 


## Disclaimer


**This repository is not OFFICIAL. If you wanna know flair itself, I recommend to look official github documentation.**


## Scripts

- src/train_model.py # how to use flair model which is the same as official documentation.
- src/use_embeddings.py # how to use pretrained model with sklearn
- main.py # FastAPI

## Usage 

Since Pipfile.lock is available, you just do 

```sh
$ pip3 install pipenv 
$ pipenv sync --dev
$ pipenv run setup
```

**flair trial**

```sh
$ pipenv run python src/train_model.py
```

**flair with sklearn**

```sh 
$ pipenv run python src/use_embeddings.py
```

**launch api**

```sh
$ pipenv run api
```
