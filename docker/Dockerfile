FROM supervisely/base-py-sdk:6.73.415

COPY dev_requirements.txt /req/dev_requirements.txt

RUN pip install --no-cache-dir -r /req/dev_requirements.txt

RUN apt-get update && apt-get install -y wireguard iproute2

WORKDIR /app

COPY src /app/src

RUN mkdir -p /app/state

EXPOSE 80

ENV PYTHONPATH="/app/src"

ENTRYPOINT ["python3", "-u", "-m", "uvicorn", "src.main:app"]
CMD ["--host", "0.0.0.0", "--port", "80"]
