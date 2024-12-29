from datetime import datetime

from requests import Session
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship

import database
from database import Base, Session

class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)

    # Define relationship to Message model
    messages = relationship("Message", back_populates="session")
    results = relationship("Result", back_populates="session")


def create_new_session(name: str):
    new_session = Session(name=name, created_at=datetime.now())
    database.Session.add(new_session)
    database.Session.commit()
    return new_session.id
