from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
engine = create_engine('sqlite:///kosen_face_db.sqlite3', echo=True)

class User(Base):
    __tablename__ = 'faces'
    id = Column(Integer, primary_key=True)
    student_name = Column(String)
    course = Column(String)
    embedding = Column(LargeBinary)

    def save_face_data(name, course, embedding):
        new_face = User(student_name=name, course=course, embedding=embedding)
        session.add(new_face)
        session.commit()

    def get_all_faces():
        return session.query(User).all()

Base.metadata.create_all(engine)