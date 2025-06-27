SHELL := /bin/bash -l

run:
	@echo "Conda環境をアクティベートしてStreamlitを実行中..."
	eval "$$(/home/lab/ryoma/miniconda3/bin/conda shell hook)" && conda activate /home/lab/ryoma/master/envs/pems-metra && streamlit run streamlit/streamlit_app.py

format:
	python format.py