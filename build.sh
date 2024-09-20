docker build --platform="linux/amd64" -t ocr-chess-game .
docker tag ocr-chess-game:latest us-central1-docker.pkg.dev/video-resume-4fcd0/cloud-run-images/ocr-chess-game:latest
docker push us-central1-docker.pkg.dev/video-resume-4fcd0/cloud-run-images/ocr-chess-game
