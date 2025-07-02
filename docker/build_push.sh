cp ../dev_requirements.txt . && \
docker build --no-cache -t supervisely/embeddings-auto-updater:0.1.0 . && \
rm dev_requirements.txt && \
docker push supervisely/embeddings-auto-updater:0.1.0
