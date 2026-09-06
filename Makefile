PYTHON ?= python
PIP    ?= $(PYTHON) -m pip
REQ    ?= requirements.txt      # RTX 50 系（sm_120）：make install REQ=requirements-cu128.txt

.PHONY: install env test train spconv-src

install:   ## 安装 requirements 中锁定的预编译 CUDA 轮子，不需要本地编译
	$(PIP) install -r $(REQ)
	$(PIP) install -e . --no-deps

env:       ## conda 方式（等价于 install，只是先建好 Python 3.11 环境）
	conda env create -f environment.yml

test:      ## 单模块形状 + 整体前向/反向（有 CUDA 用 CUDA，否则回退 CPU 冒烟）
	$(PYTHON) -m pytest -q tests

train:     ## 合成数据上的最小训练循环
	$(PYTHON) train.py

spconv-src:   ## 仅当预编译的 spconv 在你的显卡上跑不了时使用：按 spconv README 从源码 JIT 编译
	##  需要 CUDA toolkit + C++ 编译器（Windows 先在 PowerShell 里跑 spconv 仓库的 tools/msvc_setup.ps1）；
	##  RTX 50 系先 export CUMM_CUDA_ARCH_LIST="12.0"
	-$(PIP) uninstall -y spconv spconv-cu126 cumm cumm-cu126
	test -d build/cumm   || git clone https://github.com/FindDefinition/cumm build/cumm
	test -d build/spconv || git clone https://github.com/traveller59/spconv build/spconv
	$(PIP) install -e build/cumm
	$(PIP) install -e build/spconv
	$(PYTHON) -c "import spconv.pytorch"   # 首次 import 触发编译
