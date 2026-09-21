"""Fetch a list of blog posts (Medium and similar) into references/blog/<group>/ as markdown, one file per post, dated.

Medium sits behind a Cloudflare challenge for scripts, so each URL is tried in this order: the publication's RSS feed (only the
newest posts, full HTML in content:encoded), then the Wayback Machine's latest capture of the original HTML (web.archive.org
/web/2026id_/<url>), then a plain fetch. The article body is taken from the HTML's <article> element (Medium) or the main
content, converted to markdown by a small HTML walker (headings, paragraphs, lists, blockquotes, figure captions, code, links
kept as text); the date comes from the JSON-LD datePublished, the article:published_time meta tag or the RSS pubDate.

usage: uv run python scripts/fetch_blog_posts.py <links.md> <group>      writes references/blog/<group>/YYYY-MM-DD-<slug>.md
       and references/blog/<group>/INDEX.md (chronological); a line in links.md is "<url> | <title>" or a bare URL.
Every fetch is sequential with a pause; nothing is retried more than twice. Files that already exist are not refetched.
"""
import html
import json
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

from ai_experiments.paths import ROOT

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
FEEDS = {"medium.com/pinterest-engineering": "https://medium.com/feed/pinterest-engineering", "blog.zepto.com": "https://blog.zepto.com/feed",
         "medium.com/@kabirbakovic": "https://medium.com/feed/@kabirbakovic"}


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


class MD(HTMLParser):
    """HTML to markdown, enough for blog posts: block elements become paragraphs, headings keep their level, lists get bullets."""
    BLOCK = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "figcaption", "div", "section", "article", "tr", "ul", "ol", "figure", "br", "hr"}
    SKIP = {"script", "style", "nav", "header", "footer", "button", "svg", "noscript"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.buf, self.stack, self.skip = [], [], [], 0
        self.list_depth = 0

    def flush(self):
        text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        self.buf = []
        if not text:
            return
        tag = self.stack[-1] if self.stack else "p"
        if tag.startswith("h") and tag[1:].isdigit():
            self.out.append("#" * int(tag[1]) + " " + text)
        elif tag == "li":
            self.out.append("  " * max(0, self.list_depth - 1) + "- " + text)
        elif tag == "blockquote":
            self.out.append("> " + text)
        elif tag == "figcaption":
            self.out.append("*Figure: " + text + "*")
        elif tag == "pre":
            self.out.append("```\n" + "".join(self.buf_raw) + "\n```") if hasattr(self, "buf_raw") else self.out.append("    " + text)
        else:
            self.out.append(text)

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1; return
        if self.skip:
            return
        if tag in self.BLOCK:
            self.flush()
            if tag in ("ul", "ol"):
                self.list_depth += 1
            if tag in ("p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "figcaption"):
                self.stack.append(tag)
        elif tag in ("strong", "b") and not self.skip:
            self.buf.append("**")
        elif tag in ("em", "i"):
            self.buf.append("*")
        elif tag == "code" and (not self.stack or self.stack[-1] != "pre"):
            self.buf.append("`")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1); return
        if self.skip:
            return
        if tag in ("strong", "b"):
            self.buf.append("**")
        elif tag in ("em", "i"):
            self.buf.append("*")
        elif tag == "code" and (not self.stack or self.stack[-1] != "pre"):
            self.buf.append("`")
        elif tag in self.BLOCK:
            self.flush()
            if tag in ("ul", "ol"):
                self.list_depth = max(0, self.list_depth - 1)
            if self.stack and self.stack[-1] == tag:
                self.stack.pop()

    def handle_data(self, data):
        if not self.skip:
            self.buf.append(data)

    def text(self):
        self.flush()
        # collapse runs of empty emphasis and repeated blank lines
        return re.sub(r"\n{3,}", "\n\n", "\n\n".join(o for o in self.out if o.strip("*` ")))


def article_html(page):
    m = re.search(r"<article.*?</article>", page, re.S)
    return m.group(0) if m else page


def date_of(page):
    for pat in (r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})', r'article:published_time"\s+content="(\d{4}-\d{2}-\d{2})', r'"firstPublishedAt"\s*:\s*(\d{13})'):
        m = re.search(pat, page)
        if m:
            v = m.group(1)
            return time.strftime("%Y-%m-%d", time.gmtime(int(v) / 1000)) if v.isdigit() and len(v) == 13 else v
    return None


def title_of(page):
    m = re.search(r"<title[^>]*>(.*?)</title>", page, re.S)
    return html.unescape(m.group(1)).split(" | ")[0].strip() if m else ""


def feed_items(feed_url):
    try:
        xml = get(feed_url)
    except Exception as e:  # noqa: BLE001
        print(f"  feed failed: {e}"); return {}
    ns = {"content": "http://purl.org/rss/1.0/modules/content/"}
    out = {}
    for it in ET.fromstring(xml).iter("item"):
        link = (it.findtext("link") or "").split("?")[0]
        body = it.find("content:encoded", ns)
        pub = it.findtext("pubDate") or ""
        try:
            date = time.strftime("%Y-%m-%d", time.strptime(pub[:25].strip(), "%a, %d %b %Y %H:%M:%S"))
        except ValueError:
            date = None
        out[link] = dict(title=it.findtext("title") or "", html=body.text if body is not None else "", date=date)
    return out


def wayback(url):
    stamps = []
    try:  # the availability API names the closest capture; the year stamps below are the fallback
        avail = json.loads(get(f"https://archive.org/wayback/available?url={url}"))
        ts = avail.get("archived_snapshots", {}).get("closest", {}).get("timestamp")
        if ts:
            stamps.append(ts)
    except Exception as e:  # noqa: BLE001
        print(f"  availability: {e}")
    for stamp in stamps + ["2026", "2025", "2024", "2023", "2022", "2021", "2020"]:
        try:
            page = get(f"https://web.archive.org/web/{stamp}id_/{url}")
        except Exception as e:  # noqa: BLE001
            print(f"  wayback {stamp}: {e}"); time.sleep(3); continue
        if "<article" in page or "pw-post-body-paragraph" in page:
            return page
    return None


def slug_of(url):
    tail = url.rstrip("/").split("/")[-1]
    tail = re.sub(r"-[0-9a-f]{8,}$", "", tail)  # Medium's hash suffix
    return re.sub(r"[^a-z0-9-]", "", tail.lower())[:80]


def main():
    links, group = Path(sys.argv[1]), sys.argv[2]
    dest = ROOT / "references" / "blog" / group
    dest.mkdir(parents=True, exist_ok=True)
    urls = []
    for line in links.read_text().splitlines():
        m = re.match(r"\s*(https?://\S+)\s*(?:\|\s*(.*?))?\s*$", line)
        if m:
            urls.append((m.group(1).split("?")[0], (m.group(2) or "").split(" | ")[0].strip()))
    feeds = {}
    for u, _ in urls:
        for key, feed in FEEDS.items():
            if key in u and feed not in feeds:
                print(f"feed {feed}"); feeds[feed] = feed_items(feed)
    index = []
    for url, title in urls:
        slug = slug_of(url)
        existing = sorted(dest.glob(f"*-{slug}.md"))
        if existing:
            head = existing[0].read_text().splitlines()
            index.append((existing[0].name[:10], head[0].lstrip("# ").strip(), url, existing[0].name)); print(f"have {existing[0].name}"); continue
        print(f"fetch {url}")
        page = date = None; src = ""
        for feed in feeds.values():
            if url in feed and feed[url]["html"]:
                page, date, src = feed[url]["html"], feed[url]["date"], "rss"
                title = title or feed[url]["title"]
        if page is None:
            page = wayback(url); src = "wayback"
            if page is None:
                try:
                    page = get(url); src = "direct"
                except Exception as e:  # noqa: BLE001
                    print(f"  direct: {e}")
        if page is None or "Just a moment" in page[:2000]:
            print("  MISSING"); index.append(("MISSING", title, url, "")); time.sleep(2); continue
        date = date or date_of(page) or "undated"
        title = title or title_of(page)
        p = MD(); p.feed(article_html(page)); body = p.text()
        name = f"{date}-{slug}.md"
        (dest / name).write_text(f"# {title}\n\n*Source: {url}  \nPublished: {date}  \nFetched: {time.strftime('%Y-%m-%d')} via {src}*\n\n{body}\n")
        print(f"  -> {name} ({len(body.split())} words)")
        index.append((date, title, url, name)); time.sleep(4)
    index.sort()
    lines = [f"# {group}: posts in chronological order", "", "Fetched by `scripts/fetch_blog_posts.py` from the list in `references/blog/`; one file per post, dated by its publication date.", ""]
    lines += [f"- {d} [{t}]({n}) ({u})" if n else f"- MISSING {t} ({u})" for d, t, u, n in index]
    (dest / "INDEX.md").write_text("\n".join(lines) + "\n")
    print(f"index: {len(index)} entries, {sum(1 for d, *_ in index if d == 'MISSING')} missing")


if __name__ == "__main__":
    main()
