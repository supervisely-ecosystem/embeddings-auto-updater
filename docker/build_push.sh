cp ../dev_requirements.txt . && \
docker build --no-cache -t supervisely/embedding-services:0.1.0 . && \
rm dev_requirements.txt && \
docker push supervisely/embedding-services:0.1.0
