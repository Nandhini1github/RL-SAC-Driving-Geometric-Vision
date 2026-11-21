FROM python:3.12-slim
WORKDIR /driving
COPY requirements*.txt ./
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements-control.txt
COPY geometric_vision geometric_vision
COPY driving_environment driving_environment
COPY sac_training sac_training
COPY agentic_supervisor agentic_supervisor
COPY evaluation evaluation
CMD ["python", "-m", "evaluation.stereo_benchmark"]
