from requests import Session
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from database import Base, Session

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)

    # Define relationship to Message model
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")


def add_user(new_user):
    Session.add(new_user)
    Session.commit()

def get_user_by_email(email):
    user = Session.query(User).filter(User.email == email).first()
    return user
