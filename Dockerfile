# Static site: the demo runs entirely in the visitor's browser, nginx only serves files.
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm tar && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY cvlab ./cvlab
COPY demo ./demo
COPY scripts ./scripts
RUN python scripts/build_demo.py

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/site /usr/share/nginx/html
EXPOSE 80
