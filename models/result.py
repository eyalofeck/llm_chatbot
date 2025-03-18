from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship, scoped_session

import database
from database import Base, SessionLocal
from models.user import get_user_by_email


# Initialize SQLAlchemy base
# Base = declarative_base()


# Define the Message model
class Result(Base):
    __tablename__ = 'results'

    id = Column(Integer, primary_key=True, autoincrement=True)  # Unique ID for each message
    summarize = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)  # Timestamp when the message is created

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Define relationship to User model
    user = relationship("User", back_populates="results")

    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    session = relationship("Session", back_populates="results")



# def save_result(summarize, timestamp, email, session_id):
#     the_user = get_user_by_email(email)
#     new_message = Result(summarize=summarize, timestamp=timestamp, user=the_user, session_id=session_id)
#     database.Session.add(new_message)
#     database.Session.commit()


def save_result(summarize, timestamp, email, session_id):
    the_user = get_user_by_email(email)
    dbsession = scoped_session(SessionLocal)
    try:
        results = Result(summarize=summarize, timestamp=timestamp, user=the_user, session_id=session_id)
        dbsession.add(results)
        dbsession.commit()
    # except:
    # EO Debug
    except Exception as e:
        print(f"Error writing reults to DB for user: {the_user}: {e}")
        dbsession.rollback()
        raise
    finally:
        dbsession.remove()

