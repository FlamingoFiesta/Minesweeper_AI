
FROM python:3.8-slim


ENV DISPLAY=host.docker.internal:0.0


WORKDIR /usr/src/app


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY . .


CMD ["python", "./python-tkinter-minesweeper-master/minesweeper.py"]
