# models.py

from .__init__ import db 
class Trashlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(20), nullable=False)
    create_date = db.Column(db.DateTime, nullable=False)
