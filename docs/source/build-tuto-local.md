
``` shell
git clone --no-single-branch --depth 50 https://github.com/openspyrit/spyrit .
git checkout --force origin/gallery
git clean -d -f -f
cat .readthedocs.yml
```

# Linux
``` shell
python3.7 -mvirtualenv $READTHEDOCS_VIRTUALENV_PATH
python -m pip install --upgrade --no-cache-dir pip setuptools
python -m pip install --upgrade --no-cache-dir pillow==5.4.1 mock==1.0.1 alabaster>=0.7,<0.8,!=0.7.5 commonmark==0.9.1 recommonmark==0.5.0 sphinx sphinx-rtd-theme readthedocs-sphinx-ext<2.3
python -m pip install --exists-action=w --no-cache-dir -r requirements.txt
cat docs/source/conf.py
python -m sphinx -T -E -b html -d _build/doctrees -D language=en . $READTHEDOCS_OUTPUT/html
```

# Windows using conda
``` powershell
conda create --name readthedoc
conda activate readthedoc
conda install pip
python.exe -m pip install --upgrade --no-cache-dir pip setuptools
pip install --upgrade --no-cache-dir pillow==10.0.0 mock==1.0.1 alabaster==0.7.13 commonmark==0.9.1 recommonmark==0.5.0 sphinx sphinx-rtd-theme readthedocs-sphinx-ext==2.2.2
cd .\myenv\spyrit\ # replace myenv by the environment in which spyrit is installed
pip install --exists-action=w --no-cache-dir -r requirements.txt
cd .\docs\source\
python -m sphinx -T -E -b html -d _build/doctrees -D language=en . html
```
