# MAX78002 deployment (CRISP-DM stage 6)

Software-level deployment with the Analog Devices toolchain. **No physical board is used**, so
everything produced here is *software verified*, never *hardware verified*.

| Folder | Content |
|---|---|
| `model_configuration/` | `ai85ecgnet.py` (model, copy to `ai8x-training/models/`), `ecg_mitbih.py` (dataset loader, copy to `ai8x-training/datasets/`), `ai85-ecgnet.yaml` (network description, copy to `ai8x-synthesis/networks/`) |
| `quantization_output/` | QAT checkpoint, quantized checkpoint, quantized evaluation log |
| `synthesis_output/` | `ai8xize.py` log and generated C project (`cnn.c`, `cnn.h`, `weights.h`, `sampledata.h`, ...) |

> The files in `model_configuration/` are **drafts that have not been run through the toolchain yet**.
> Try the full pipeline early with an untrained/briefly trained model - synthesis failures should
> feed back into the architecture (CRISP-DM iteration), not surface in the final week.

## Toolchain facts to keep the architecture compatible

From the ai8x-training/ai8x-synthesis documentation (MAX78002):

- Conv1d: kernel 1-9, padding 0-2 (padding must be 0 with more than 64 input channels), stride 1
- 1D pooling: size 1-16, stride 1-16
- Channels per layer <= 2048, layers <= 128
- Linear: <= 1024 input and <= 1024 output features
- Kernel (weight) memory 2,340 KiB, data memory 1,280 KiB
- Course limit: model size <= 2 MB

## 1. Install (Linux or WSL2)

The toolchain officially supports Ubuntu 20.04/22.04 (WSL2 works with CUDA 12.1+), Python 3.11.8, PyTorch 2.3.

```bash
# system packages + pyenv: follow the ai8x-training README
pyenv install 3.11.8

git clone --recursive https://github.com/analogdevicesinc/ai8x-training.git
git clone --recursive https://github.com/analogdevicesinc/ai8x-synthesis.git

cd ai8x-training
pyenv local 3.11.8
python -m venv .venv --prompt ai8x-training && source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
deactivate

cd ../ai8x-synthesis
pyenv local 3.11.8
python -m venv .venv --prompt ai8x-synthesis && source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

Clone the toolchain **next to** this repository (both folders are git-ignored if cloned inside it).

## 2. Copy project files

```bash
REPO=/path/to/D7043E
cp $REPO/max78002/model_configuration/ai85ecgnet.py   ai8x-training/models/
cp $REPO/max78002/model_configuration/ecg_mitbih.py   ai8x-training/datasets/
cp $REPO/max78002/model_configuration/ai85-ecgnet.yaml ai8x-synthesis/networks/
```

The datasets are the same `data/processed/*.npz` files as the FP32 pipeline (run `src/prepare_data.py` first).

## 3. Quantization-aware training (ai8x-training)

```bash
cd ai8x-training && source .venv/bin/activate
python train.py --device MAX78002 --model ai85ecgnet --dataset ECG_MITBIH --data $REPO/data/processed \
    --optimizer Adam --lr 0.001 --epochs 50 --batch-size 256 --use-bias --deterministic \
    --qat-policy policies/qat_policy.yaml --validation-split 0 --print-freq 100
```

**`--validation-split 0` is mandatory**: a non-zero value takes a random beat-level split of the
training data. With 0, the `ECG_MITBIH` "test" loader (= DS1 validation records) is used for
validation. Check the training log to confirm this behaviour in the installed version.

## 4. Quantize and evaluate (ai8x-synthesis / ai8x-training)

```bash
cd ai8x-synthesis && source .venv/bin/activate
python quantize.py ../ai8x-training/logs/<run>/qat_best.pth.tar \
    $REPO/max78002/quantization_output/ai85-ecgnet-q.pth.tar --device MAX78002 -v

cd ../ai8x-training && source .venv/bin/activate
# INT8 evaluation on the validation records; also saves the sample input used by synthesis
python train.py --device MAX78002 --model ai85ecgnet --dataset ECG_MITBIH --data $REPO/data/processed \
    --evaluate --exp-load-weights-from $REPO/max78002/quantization_output/ai85-ecgnet-q.pth.tar \
    -8 --use-bias --save-sample 10
```

Final DS2 numbers: repeat the evaluation with `--dataset ECG_MITBIH_DS2`, only once the model is frozen.

## 5. Synthesize C code

```bash
cd ai8x-synthesis && source .venv/bin/activate
python ai8xize.py --device MAX78002 --prefix ai85-ecgnet \
    --checkpoint-file $REPO/max78002/quantization_output/ai85-ecgnet-q.pth.tar \
    --config-file networks/ai85-ecgnet.yaml \
    --sample-input ../ai8x-training/sample_ecg_mitbih.npy \
    --test-dir $REPO/max78002/synthesis_output \
    --softmax --compact-data --mexpress --timer 0 --display-checkpoint --verbose --overwrite \
    2>&1 | tee $REPO/max78002/synthesis_output/ai8xize.log
```

Record weight and data memory from the log for the deployment-evidence table (report section 6.3).
Verify all command-line flags against `python <script>.py --help` of the installed toolchain version.
