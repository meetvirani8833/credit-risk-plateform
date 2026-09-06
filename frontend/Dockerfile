# Multi-stage: build the Vite SPA, then serve the static output with nginx.
FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Baked in at build time (Vite env vars are compile-time, not runtime),
# defaults to the backend's docker-compose service reachable from the
# host browser at localhost:8000, override with --build-arg for a
# different backend URL.
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
