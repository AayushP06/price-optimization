from flask_sqlalchemy import SQLAlchemy

# Single shared SQLAlchemy instance — imported by models and routes
db = SQLAlchemy()
