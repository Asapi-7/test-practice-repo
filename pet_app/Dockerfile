FROM python:3.10.11

WORKDIR /pet_app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY ./backend/www /pet_app/www

# hostとportを明示的に設定 (0.0.0.0はコンテナ外からのアクセスを許可するために必要)
#上は機械学習モデルを使用する際に使う、下はモデルを使用しない際に使う　使わない方はコメントアウトすること！何が起こるかわからんので
CMD ["uvicorn", "backend.backendTest:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["uvicorn", "backend.backend:app", "--host", "0.0.0.0", "--port", "8000"]