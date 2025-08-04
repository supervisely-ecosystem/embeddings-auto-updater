cp ../dev_requirements.txt . && \
docker build --no-cache -f Dockerfile -t supervisely/embeddings-auto-updater:0.1.2 .. && \
rm dev_requirements.txt && \
docker push supervisely/embeddings-auto-updater:0.1.2
