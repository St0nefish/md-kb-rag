---
title: Deploying with Docker Compose
description: Step-by-step guide to deploying services with Docker Compose on a Linux host.
type: guide
domain: infrastructure
tags:
  - docker
  - deployment
  - linux
status: active
---

## Deploying with Docker Compose

Docker Compose lets you define and run multi-container applications with a single YAML file.

## Prerequisites

- Docker Engine 24+ installed
- Docker Compose v2 plugin
- A Linux host with at least 2 GB RAM

## Writing the Compose File

Create a `docker-compose.yml` in your project root:

```yaml
services:
  app:
    image: myapp:latest
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgres://db:5432/myapp
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready"]
      interval: 10s

volumes:
  pgdata:
```

## Starting the Stack

```bash
docker compose up -d
docker compose logs -f app
```

## Updating Services

Pull new images and recreate only changed containers:

```bash
docker compose pull
docker compose up -d
```
