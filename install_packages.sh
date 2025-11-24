#!/bin/bash
pip install --upgrade pip

pip install langchain-core==0.3.79 --force-reinstall
pip install langchain==0.3.0 --force-reinstall
pip install langchain-community==0.3.0 --force-reinstall
pip install "langchain-weaviate>=0.0.6,<0.1.0" --force-reinstall
pip install "weaviate-client>=3.0.0,<4.0.0" --force-reinstall
pip install "langchain-mistralai>=0.2.0,<1.0.0" --force-reinstall
pip install langchain-huggingface==0.3.0 --force-reinstall
pip install streamlit --force-reinstall
pip install python-dotenv --force-reinstall
pip install mistralai --force-reinstall
pip install pypdf --force-reinstall
pip install sentence-transformers --force-reinstall
pip install pdf2image --force-reinstall
pip install python-docx --force-reinstall
pip install fpdf --force-reinstall
pip install opencv-python-headless --force-reinstall
pip install pytesseract --force-reinstall
pip install bert-score --force-reinstall
pip install nltk --force-reinstall
pip install scikit-learn --force-reinstall

echo "Все пакеты установлены"