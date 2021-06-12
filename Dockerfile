FROM python:3.9.5

# set working directory
WORKDIR /usr

# virtualenv
#ENV VIRTUAL_ENV=/opt/venv
#RUN python3 -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# add and install requirements
RUN pip install --upgrade pip
COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt
RUN pip3 install streamlit


# Path
#ENV PATH="/opt/venv/bin:$PATH"

RUN mkdir /src
COPY ./src /src
COPY ./main.py /
COPY . .

#ENTRYPOINT [ "streamlit run" ]
#RUN streamlit run
#CMD [ "main.py" ]
CMD streamlit run main.py

EXPOSE 8501
