FROM akkadeeemikk/mats

RUN pip install uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
RUN pip install datasets==3.6.0
RUN pip install vastai
RUN pip install --no-cache-dir --force-reinstall "clearml>=2.0.2"
