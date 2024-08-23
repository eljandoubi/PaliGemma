<p align="center">
    <a href="docs/imgs/PaLiGemma-Model-Logo.jpg">
        <img src="docs/imgs/PaLiGemma-Model-Logo.jpg" alt="Jumanji logo" width="50%"/>
    </a>
</p>

Coding PaliGemma from scratch using pytorch for inference.

## Setup environment
* Clone the repository and Go to PaliGemma directory.
```bash
git clone https://github.com/eljandoubi/PaliGemma.git & cd PaliGemma
```

* Build environment.
```bash
make build
```

* Run inference.
```bash
make run
```

* Clean environment.
```bash
make clean
```

## Costumized tests
You can change these variables: `PROMPT` and `IMAGE_FILE_PATH` in order to run on your own test case.
```bash
make run PROMPT="this building is " IMAGE_FILE_PATH="sample/EiffelTower.jpg"
```
