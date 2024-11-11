FROM alpine:latest

ENV PYTHONUNBUFFERED=1
RUN apk add --no-cache python3
RUN apk add --no-cache py3-pip
RUN pip3 install --break-system-packages Flask
RUN pip3 install --break-system-packages gunicorn
RUN pip3 install --break-system-packages cloudevents
RUN pip3 install --break-system-packages openai
RUN pip3 install --break-system-packages python-dotenv
RUN pip3 install --break-system-packages boto3
RUN pip3 install --break-system-packages google-cloud-documentai


COPY main.py .
COPY amazon.py .
COPY .env .

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
