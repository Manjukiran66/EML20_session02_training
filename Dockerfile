FROM python:3.8.14-slim-bullseye


WORKDIR /opt/src

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt && rm -rf /root/.cahe/pip

ADD eml20_session02_training  /opt/src/eml20_session02_training

ADD config /opt/src/config

ADD entrypoint.sh entrypoint.sh

ADD pyproject.toml pyproject.toml 


ENTRYPOINT ["sh","entrypoint.sh"]
