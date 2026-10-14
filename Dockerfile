# Static site: the demo runs entirely in the visitor's browser, nginx only serves files.
FROM node:22-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends python3 tar && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY cvlab ./cvlab
COPY demo ./demo
COPY scripts ./scripts
RUN python3 scripts/build_demo.py

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/site /usr/share/nginx/html
EXPOSE 80
