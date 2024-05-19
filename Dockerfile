FROM python:3.8-slim

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 9201

ENV NAME mnist_env

CMD ["python", "app.py"]
