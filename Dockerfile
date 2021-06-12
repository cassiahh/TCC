FROM python:3.9

COPY ./requirements.txt /requirements.txt

RUN pip3 install -r requirements.txt

RUN mkdir /src
COPY ./src /src
COPY ./main.py /

#ENTRYPOINT [ "streamlit run" ]
#CMD [ "main.py" ]
CMD streamlit run main.py

EXPOSE 8501
