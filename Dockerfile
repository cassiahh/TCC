FROM python:3.9.5

EXPOSE 8501

WORKDIR /

ENV PYTHONUNBUFFERED 1
# set working directory
#WORKDIR /usr

## virtualenv
#ENV VIRTUAL_ENV=/opt/venv
#RUN python3 -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# add and install requirements
RUN pip install --upgrade pip
#COPY ./requirements.txt ./
COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt --no-cache-dir
#RUN pip3 install -r requirements.txt
#RUN pip3 install streamlit

## PATH
#ENV PATH="/opt/venv/bin:$PATH"

#RUN mkdir /src
COPY main.py ./main.py
COPY src /src

#COPY . .

#ENTRYPOINT [ "/bin/bash", "streamlit", "run" ]
##ENTRYPOINT [ "streamlit", "run" ]
###RUN streamlit run
##CMD [ "main.py" ]
#
#CMD [ "main.py" ]

CMD streamlit run main.py
