FROM pytorch/pytorch

# PYTHONUNBUFFERED, prints logs immediately, not buffered
# PYTHONDONTWRITEBYTECODE, stops Python from writing .pyc compiled files inside the container
ENV PYTHONUNBUFFERED=1 \    
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]