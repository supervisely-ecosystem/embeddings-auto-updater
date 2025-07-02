FROM supervisely/base-py-sdk:6.73.350

COPY dev_requirements.txt /req/dev_requirements.txt

RUN pip install --no-cache-dir -r /req/dev_requirements.txt

WORKDIR /app

COPY src /app/src

RUN mkdir -p /app/state

EXPOSE 80

ENTRYPOINT ["uvicorn", "src.main:app"]
CMD ["--host", "0.0.0.0", "--port", "80"]