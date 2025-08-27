FROM akkadeeemikk/mats

RUN pip install uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
RUN pip install datasets==3.6.0
