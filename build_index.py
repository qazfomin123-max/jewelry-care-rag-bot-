name: Build RAG Index

on:
  push:
    branches: [ main ]
    paths:
      - 'knowledge_base/**'
      - 'build_index.py'
      - 'requirements.txt'
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build-index:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Build index
        run: python build_index.py

      - name: Commit and push chroma_db
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add chroma_db
          git diff --staged --quiet && echo "No changes to commit" || git commit -m "Auto-rebuild RAG index [skip ci]"
          git push
