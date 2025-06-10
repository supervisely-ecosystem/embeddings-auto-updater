cp ../dev_requirements.txt . && \
docker build --no-cache -t supervisely/embeddings-generator:0.1.3rc7 . && \
rm dev_requirements.txt && \
docker push supervisely/embeddings-generator:0.1.3rc7 
