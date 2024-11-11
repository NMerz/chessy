docker build --platform="linux/amd64" -f Dockerfile_play -t play-chess ./
docker tag play-chess:latest us-central1-docker.pkg.dev/video-resume-4fcd0/cloud-run-images/play-chess:latest
docker push us-central1-docker.pkg.dev/video-resume-4fcd0/cloud-run-images/play-chess
