FROM waujito/nto23_1_env:1.4
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["celery", "-A", "tasks", "worker", "--loglevel=INFO"]
