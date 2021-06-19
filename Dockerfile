FROM python:3.9.5

EXPOSE 8501

#WORKDIR /usr

## virtualenv
#ENV VIRTUAL_ENV=/opt/venv
#RUN python3 -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# add and install requirements
RUN pip install --upgrade pip
COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt --no-cache-dir

## PATH
#ENV PATH="/opt/venv/bin:$PATH"

COPY . .
COPY main.py ./main.py
COPY src /src

##ENTRYPOINT [ "streamlit" ]
##CMD [ "run", "main.py" ]

CMD streamlit run main.py


