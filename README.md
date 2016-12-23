# tf_dl_lab_ex4

This is a course assignment for a deep learning course at university of Freiburg.
Implementation hyperband and random search for optimization of convolutional neural network on MNIST dataset.

## Setup  

Install setuptools (Mac OS X)

```
curl https://bootstrap.pypa.io/ez_setup.py -o - | python
```

Install requirements

```
pip install -r requirements.txt
```

## Execution

Run hyperband parameter optimization
```
python master_hyperband.py
```

Run random search parameter optimization
```
python master_random.py
```

## Parallelization 

You can run hyperband and random search in parallel for faster optimization.
