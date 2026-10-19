
# Runtime image for the ood-mt Open OnDemand app
# (Streamlit front end + NLLB-200 via Hugging Face transformers).
#
# template/script.sh.erb runs it as:
#   singularity exec --cleanenv --nv <image> python -m streamlit run app.py ...
# so this image only provides Python and the libraries. The app code is not
# baked in: OOD stages template/ into the job directory and app.py is run
# from there.
#
# Build (from the repo root):
#   docker build -t ood-mt:latest -f container/Dockerfile container/
# Convert for the cluster:
#   apptainer build ood-mt.sif docker-daemon://ood-mt:latest
#   (or push to a registry and use docker://<registry>/ood-mt:latest)
 
FROM python:3.13-slim
 
# Anything the app needs at runtime has to be set here, not in script.sh.erb:
# --cleanenv drops the job's environment before python starts.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
 
# PyTorch with CUDA 12.6, in its own layer so dependency edits don't re-download it.
# Why cu126 and not the default PyPI wheel (CUDA 13):
#   - CUDA 13 builds don't support Pascal, so the form's P100 option would fail;
#     2.14 is the last PyTorch release that still ships a cu126 build.
#   - cu126 runs on any host driver >= 525; CUDA 13 needs >= 580.
# The CUDA runtime libraries come with the wheel; the driver comes from the
# host through --nv, so no nvidia/cuda base image is needed.
RUN pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1
 
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt \
 && pip check \
 && python -c "import torch; assert torch.version.cuda.startswith('12.6'), torch.version.cuda; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"
 
# Handy when testing the image directly with docker run; OOD does not use it.
EXPOSE 8501
CMD ["python", "-m", "streamlit", "--version"]