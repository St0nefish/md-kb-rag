---
name: Bug report
about: Something is broken or behaving unexpectedly
title: ""
labels: bug
assignees: ""
---

## Problem

What's wrong? Include the exact error message, log line, or unexpected behavior. A
pointer to the relevant file/line if you've found one (`src/foo.rs:123`) is
appreciated but not required.

## Failure scenario

How does this actually manifest — what has to happen for you to hit it? Steps to
reproduce, or the sequence of events that led here, help far more than a general
description.

## Environment

- mcp-md-wiki version / image tag (or commit sha if built from source):
- Deployment: Docker Compose / bare binary / other:
- Relevant config (redact secrets): chunking, embedding, search, or reranking
  settings if the bug is retrieval- or indexing-related.

## Suggested fix (optional)

If you have a guess at the cause or a fix in mind, note it here. Not required — a
clear problem statement is the useful part.
