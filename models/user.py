# from requests import Session
from math import trunc

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship, scoped_session
from database import Base, SessionLocal


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)

    # Define relationship to Message model
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
    results = relationship("Result", back_populates="user", cascade="all, delete-orphan")


# def add_user(new_user, user_email):
#     if get_user_by_email(user_email):
#         return
#     Session.add(new_user)
#     Session.commit()


def add_user(new_user, user_email):
    dbsession = scoped_session(SessionLocal)
    try:
        if get_user_by_email(user_email, dbsession):
            return
        dbsession.add(new_user)
        dbsession.commit()
    except:
        dbsession.rollback()
        raise
    finally:
        dbsession.remove()

# def get_user_by_email(email):
#     user = Session.query(User).filter(User.email == email).first()
#     return user

def get_user_by_email(email, dbsession=None):
    created_here = False
    if dbsession is None:
        dbsession = scoped_session(SessionLocal)
        created_here = True
    try:
        user = dbsession.query(User).filter(User.email == email).first()
        return user
    # EO Debug
    except Exception as e:
        print(f"Error fetching user {email}: {e}")
        raise
    finally:
        if created_here:
            dbsession.remove()

