# PaliGemma

Coding PaliGemma from scratch using pytorch for inference.

## Setup environment
* Clone the repository.
```bash
git clone https://github.com/eljandoubi/PaliGemma.git
```
* Go to PaliGemma directory.
```bash
cd PaliGemma
```
* Make virtual environment.
```bash
conda create -n PG python=3.12
conda acctivate PG
```
* Install dependencies.
```bash
pip install -r requirements.txt
```
* Make weights folder.

```bash
mkdir -p $HOME/paligemma-weights/paligemma-3b-pt-224
```

* Download PaliGemma weights from [paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224/tree/main) to `paligemma-3b-pt-224` folder.

* Change the mode of `run_infer.sh` to execution.
```bash
chmod +x run_infer.sh
```

* Run inference.
```bash
./run_infer.sh
```




