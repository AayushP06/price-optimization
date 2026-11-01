FROM node:20-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY ml-model/xgb_model.json ml-model/feature_columns.json ./ml-model/
COPY --from=frontend-build /frontend/dist ./frontend/dist
WORKDIR /app/backend
ENV PORT=5000
EXPOSE 5000
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} app.main:app"]
