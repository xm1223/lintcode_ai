## Install pyenv on Mac
https://github.com/pyenv/pyenv

```
$ git clone https://github.com/pyenv/pyenv.git ~/.pyenv
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
$ echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
```
## Install python with certain version, i.e. 3.7.2 and SciPy
```
$ pyenv install 3.7.2
$ pyenv global 3.7.2
$ pip install numpy
$ pip install pandas
$ pip install scipy
$ pip install scikit-learn
$ pip install  nltk
$ python -m pip install jupyter
```

## Setup PyCharm
https://www.jetbrains.com/pycharm/download

