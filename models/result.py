from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

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




# Example: Save a new message
def save_result(summarize, timestamp, email):
    the_user = get_user_by_email(email)
    new_message = Result(summarize=summarize, timestamp=timestamp, user=the_user)
    database.Session.add(new_message)
    database.Session.commit()

