# Gunicorn configuration file
workers = 4
timeout = 120  # 2 minutes timeout
worker_class = 'sync'
bind = '0.0.0.0:5000'