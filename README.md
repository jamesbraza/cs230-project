# cs230-project

[Stanford CS230](https://cs230.stanford.edu/): Class Project

## Developers

This project was developed using Python 3.10.

### Getting Started With `pre-commit`

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-qa.txt
pre-commit install
```

### Debugging with `tensorboard`

Here is how you kick off `tensorboard`:

```bash
tensorboard --logdir training
```

Afterwards, go to its URL: [http://localhost:6006/](http://localhost:6006/).
