#!/usr/bin/env python3
"""Overlay the hand-written site and its legacy URLs on a Hugo build."""
from pathlib import Path
import re
import shutil
import sys

prototype = Path(__file__).resolve().parent
blog_source = prototype.parent
site = blog_source.parent.parent
public = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else site / 'public'
if not public.is_dir():
    raise SystemExit('Run hugo before deploying the static pages.')

def copy_directory(source, destination):
    for item in source.rglob('*'):
        if item.is_file():
            target = destination / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


for item in prototype.iterdir():
    if item.name in ('.gitignore', 'deploy.py'):
        continue
    destination = public / item.name
    if item.is_dir():
        copy_directory(item, destination)
    else:
        shutil.copy2(item, destination)

# Keep a single copy of each existing photo and PDF in the repository.
for source in blog_source.iterdir():
    if not source.is_dir() or source.name.startswith('_'):
        continue
    destination = public / 'posts' / source.name
    if not destination.is_dir():
        continue
    for asset in source.iterdir():
        if asset.is_dir() and asset.name == 'img':
            copy_directory(asset, destination / asset.name)
        elif asset.is_file() and asset.suffix.lower() == '.pdf':
            shutil.copy2(asset, destination / asset.name)

# Keep old article URLs serving the same plain HTML, without redirects.
for article in (public / 'posts').glob('*/*.html'):
    slug, language = article.parent.name, article.stem
    if language not in ('en', 'zh'):
        continue
    old = public / ('zh/blog' if language == 'zh' else 'blog') / slug
    old.mkdir(parents=True, exist_ok=True)
    content = article.read_text()
    content = content.replace('href="../../index.html"', 'href="/"')
    content = content.replace('href="../../style.css"', 'href="/style.css"')
    content = re.sub(r'(src|href)="(img/[^"#?]+|[^"/]+\.pdf)([?#][^"]*)?"',
                     lambda m: '{}="/posts/{}/{}{}"'.format(m[1], slug, m[2], m[3] or ''), content)
    content = content.replace('href="en.html"', 'href="/blog/{}/"'.format(slug))
    content = content.replace('href="zh.html"', 'href="/zh/blog/{}/"'.format(slug))
    (old / 'index.html').write_text(content)
print('Published static pages and existing article URLs to', public)
