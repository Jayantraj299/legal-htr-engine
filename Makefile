# Legal-HTR Engine Makefile
SHELL := /bin/bash
PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

# Configurable runtime parameters
DOC ?= data/processed/tight_lines_test.png
OUT ?= data/processed/test_transcription.txt
WORK_DIR ?= data/processed/pipeline_debug
BATCH_SIZE ?= 4
WINDOW_SIZE ?= 25
SAUVOLA_K ?= 0.2
BENCHMARK ?= all

.DEFAULT_GOAL := help

.PHONY: help setup ingest ingest-iam ingest-bentham run binarize segment infer eval eval-quick dvc-track clean

help:
	@echo "=================================================================="
	@echo " Legal-HTR Engine Build & Execution Targets"
	@echo "=================================================================="
	@echo "  make setup           Create .venv and install dependencies"
	@echo "  make ingest          Ingest both IAM and Bentham benchmark splits"
	@echo "  make run             Run full HTR pipeline on a document"
	@echo "                       (Override: make run DOC=path/to/scan.png OUT=out.txt)"
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
	    --output "$(OUT)" \
	    --work-dir "$(WORK_DIR)" \
	    --batch-size $(BATCH_SIZE) \
	    --window-size $(WINDOW_SIZE) \
	    -k $(SAUVOLA_K)

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
	    --benchmark $(BENCHMARK) \nannano 
	    --batch-size $(BATCH_SIZE) \
	    --max-samples 5

dvc-track:
	dvc add data

clean:
	rm -rf $(WORK_DIR) data/processed/run*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
