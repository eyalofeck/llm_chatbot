from datetime import datetime

#from requests import Session
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship, scoped_session, sessionmaker

import database
from database import Base, engine

SessionFactory = sessionmaker(bind=engine)
SessionLocal = scoped_session(SessionFactory)

class ChatSession(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)

    # Define relationship to Message model
    messages = relationship("Message", back_populates="session")
    results = relationship("Result", back_populates="session")


def create_new_session(name: str):
    db_session = SessionLocal()
    new_session = ChatSession(name=name, created_at=datetime.now())
    db_session.add(new_session)
    db_session.commit()
    db_session.close()
    return new_session.id
