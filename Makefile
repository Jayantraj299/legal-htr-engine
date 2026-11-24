# Legal-HTR Engine Makefile
SHELL := /bin/bash
PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

# Configurable runtime parameters
DOC ?= data/raw/real_samples/aviation.jpg
OUT ?= data/processed/aviation_readable
OUT_BASE := $(basename $(OUT))
WORK_DIR ?= data/processed/run_doc
MODEL ?= microsoft/trocr-base-handwritten
BATCH_SIZE ?= 4
WINDOW_SIZE ?= 25
SAUVOLA_K ?= 0.2
BENCHMARK ?= all

# Digital text / PDF context analysis parameters
TEXT_IN ?= data/raw/real_samples
TEXT_OUT ?= data/processed/text_contexts

.DEFAULT_GOAL := help

.PHONY: help setup ingest ingest-iam ingest-bentham \
	    run run-classic run-printed run-embed process-text \
	    view open binarize segment infer eval eval-quick dvc-track clean

help:
	@echo "=================================================================="
	@echo " Legal-HTR Engine Build & Execution Targets"
	@echo "=================================================================="
	@echo "  make setup           Create .venv and install CUDA PyTorch + deps"
	@echo "  make ingest          Ingest IAM and Bentham benchmark splits"
	@echo "  make run             Run archival cleaner + HTR + Legal-BERT (.md/.html/.json)"
	@echo "                       (Override: make run DOC=path/to/scan.jpg OUT=data/processed/doc)"
	@echo "  make run-classic     Run classic Sauvola + HPP seam-carving pipeline"
	@echo "  make run-printed     Run pipeline using trocr-base-printed model"
	@echo "  make run-embed       Run pipeline and export full 768-d pgvector embedding"
	@echo "  make process-text    Run Legal-BERT context analysis on .txt/.md/.pdf files"
	@echo "                       (Override: make process-text TEXT_IN=path/to/file.pdf)"
	@echo "  make view            Print the generated readable Markdown document"
	@echo "  make open            Open the styled HTML document in Brave Browser"
	@echo "  make eval            Evaluate CER & WER across full test splits"
	@echo "  make eval-quick      Quick 5-sample CER & WER smoke test"
	@echo "  make binarize        Run standalone Sauvola adaptive binarization"
	@echo "  make segment         Run standalone HPP + Seam-Carving segmentation"
	@echo "  make infer           Run standalone FP16 TrOCR batch inference"
	@echo "  make dvc-track       Update DVC tracking manifest for data/"
	@echo "  make clean           Remove temporary processed runs and caches"
	@echo "=================================================================="

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install dvc

ingest: ingest-iam ingest-bentham

ingest-iam:
	$(PYTHON) scripts/ingest_iam.py

ingest-bentham:
	$(PYTHON) scripts/ingest_bentham.py

run:
	$(PYTHON) src/pipeline.py \
	    --input "$(DOC)" \
	    --output "$(OUT_BASE)" \
	    --work-dir "$(WORK_DIR)" \
	    --model "$(MODEL)" \
	    --batch-size $(BATCH_SIZE)

run-classic:
	$(PYTHON) src/pipeline.py \
	    --input "$(DOC)" \
	    --output "$(OUT_BASE)" \
	    --work-dir "$(WORK_DIR)" \
	    --model "$(MODEL)" \
	    --batch-size $(BATCH_SIZE) \
	    --classic-sauvola \
	    --window-size $(WINDOW_SIZE) \
	    -k $(SAUVOLA_K)

run-printed:
	$(PYTHON) src/pipeline.py \
	    --input "$(DOC)" \
	    --output "$(OUT_BASE)" \
	    --work-dir "$(WORK_DIR)" \
	    --model "microsoft/trocr-base-printed" \
	    --batch-size $(BATCH_SIZE)

run-embed:
	$(PYTHON) src/pipeline.py \
	    --input "$(DOC)" \
	    --output "$(OUT_BASE)" \
	    --work-dir "$(WORK_DIR)" \
	    --model "$(MODEL)" \
	    --batch-size $(BATCH_SIZE) \
	    --full-embedding

process-text:
	$(PYTHON) src/analysis/process_text.py \
	    --input "$(TEXT_IN)" \
	    --output-dir "$(TEXT_OUT)" \
	    --full-embedding

view:
	@cat "$(OUT_BASE).md"

open:
	@brave "$(OUT_BASE).html" >/dev/null 2>&1 &

binarize:
	@mkdir -p $(WORK_DIR)
	$(PYTHON) src/preprocessing/binarize.py \
	    --input "$(DOC)" \
	    --output "$(WORK_DIR)/binarized.png" \
	    --window-size $(WINDOW_SIZE) \
	    -k $(SAUVOLA_K)

segment:
	@mkdir -p $(WORK_DIR)/lines
	$(PYTHON) src/preprocessing/line_segment.py \
	    --input "$(WORK_DIR)/binarized.png" \
	    --output-dir "$(WORK_DIR)/lines"

infer:
	$(PYTHON) src/engine/model.py \
	    --input-dir "$(WORK_DIR)/lines" \
	    --output "$(WORK_DIR)/predictions.json" \
	    --batch-size $(BATCH_SIZE)

eval:
	$(PYTHON) src/engine/evaluate.py \
	    --benchmark $(BENCHMARK) \
	    --batch-size $(BATCH_SIZE)

eval-quick:
	$(PYTHON) src/engine/evaluate.py \
	    --benchmark $(BENCHMARK) \
	    --batch-size $(BATCH_SIZE) \
	    --max-samples 5

dvc-track:
	dvc add data

clean:
	rm -rf $(WORK_DIR) data/processed/run* data/processed/pipeline_debug
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
