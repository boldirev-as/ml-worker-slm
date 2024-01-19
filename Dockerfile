COPY . .
CMD ["celery", "-A", "tasks", "worker", "--loglevel=INFO"]
