"""Move settled Hugo page-bundle media to a verified GitHub release."""

import collections
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlencode, urljoin, urlsplit

MEDIA_TYPES = {".gif": "image/gif", ".mp4": "video/mp4", ".webm": "video/webm"}
TEXT_TYPES = {".md", ".html", ".json", ".js", ".css", ".yaml", ".yml", ".toml"}
WAIT = dt.timedelta(days=7)
ROLLOUT = dt.datetime(2026, 11, 12, tzinfo=dt.timezone.utc)
BRANCH_PREFIX = "automation/blog-media-"
REPO = os.environ.get("GITHUB_REPOSITORY", "hao-ai-lab/hao-ai-lab.github.io")
DEFAULT_BRANCH = os.environ.get("DEFAULT_BRANCH", "main")
SAFE_PATH = re.compile(r"[A-Za-z0-9_.\-/]+\Z")


@dataclass(frozen=True)
class Asset:
    path: Path
    public_path: str
    name: str
    size: int
    sha256: str


def command(*args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def api(path, *args):
    raw = command("gh", "api", f"repos/{REPO}/{path}", *args)
    return json.loads(raw) if raw else None


def timestamp(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def settled_production(cutoff):
    params = urlencode({
        "branch": DEFAULT_BRANCH, "event": "push", "status": "success", "per_page": 10,
        "created": "<=" + cutoff.isoformat().replace("+00:00", "Z"),
    })
    runs = api(f"actions/workflows/hugo.yaml/runs?{params}")["workflow_runs"]
    eligible = [run for run in runs if timestamp(run["updated_at"]) <= cutoff]
    if not eligible:
        return {}, cutoff
    run = max(eligible, key=lambda item: item["updated_at"])
    tree = api(f"git/trees/{run['head_sha']}?recursive=1")
    if tree["truncated"]:
        raise RuntimeError("The production Git tree was truncated")
    blobs = {Path(entry["path"]): entry["sha"] for entry in tree["tree"] if entry["type"] == "blob"}
    return blobs, timestamp(run["updated_at"])


def published_posts(rows, now):
    for row in rows:
        index = Path(row["path"])
        published = timestamp(row["publishDate"])
        expires = timestamp(row["expiryDate"])
        if (
            len(index.parts) == 4
            and index.parts[:2] == ("content", "blogs")
            and index.name == "index.md"
            and row["draft"].lower() == "false"
            and ROLLOUT <= published <= now - WAIT
            and (expires.year == 1 or expires > now)
        ):
            yield index, row["permalink"], published


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def candidates(tracked, now):
    rows = csv.DictReader(io.StringIO(command("hugo", "list", "all", "--environment", "production")))
    posts = [(index, url, published) for index, url, published in published_posts(rows, now) if index in tracked]
    if not posts:
        return
    deployed, deployed_at = settled_production(now - WAIT)
    for index, permalink, published in posts:
        if published > deployed_at or deployed.get(index) != tracked[index]:
            continue
        for path, blob in tracked.items():
            if (
                path.is_relative_to(index.parent)
                and path.suffix.lower() in MEDIA_TYPES
                and SAFE_PATH.fullmatch(path.as_posix())
                and len(path.name) <= 230
                and not path.is_symlink()
                and deployed.get(path) == blob
            ):
                relative = path.relative_to(index.parent).as_posix()
                fingerprint = digest(path)
                yield Asset(
                    path, urlsplit(urljoin(permalink, relative)).path,
                    f"{fingerprint[:16]}-{path.name}", path.stat().st_size, fingerprint,
                )


class MediaLinks(HTMLParser):
    def __init__(self, page_url):
        super().__init__()
        self.page_url = page_url
        self.links = collections.Counter()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if not value:
                continue
            if key == "srcset":
                values = [item.split()[0] for item in value.split(",") if item.strip()]
            elif key in {"src", "href", "poster", "data-src", "content"}:
                values = [value]
            else:
                continue
            for candidate in values:
                url = urljoin(self.page_url, candidate)
                if Path(urlsplit(url).path).suffix.lower() in MEDIA_TYPES:
                    self.links[url] += 1


def build(directory, resources):
    env = dict(os.environ, HUGO_RESOURCEDIR=str(resources))
    command("hugo", "--gc", "--minify", "--environment", "production",
            "--destination", str(directory), env=env)
    files = {path.relative_to(directory) for path in directory.rglob("*") if path.is_file()}
    pages = {}
    for path in sorted(files):
        if path.suffix == ".html":
            parser = MediaLinks(urljoin("https://haoailab.com/", path.as_posix().removesuffix("index.html")))
            parser.feed((directory / path).read_text())
            pages[path] = parser.links
    return files, pages


def replace_links(text, links):
    if not links:
        return text
    choices = "|".join(re.escape(key) for key in sorted(links, key=len, reverse=True))
    pattern = re.compile(r"(?<![\w./-])(?:" + choices + r")(?![\w./-])")
    return pattern.sub(lambda match: links[match[0]], text)


def edit_sources(tracked, assets, urls):
    absolute = {}
    relative = collections.defaultdict(dict)
    for asset in assets:
        url = urls[asset.public_path]
        absolute[asset.public_path] = url
        for origin in ("https://haoailab.com", "https://hao-ai-lab.github.io"):
            absolute[origin + asset.public_path] = url
        post = Path(*asset.path.parts[:3])
        local = asset.path.relative_to(post).as_posix()
        relative[post].update({local: url, "./" + local: url})
    video_tag = re.compile(
        r'(<video\b[^>]*?)\s+src=(["\'])(https://[^"\']+/releases/download/blog-media-[^"\']+)\2([^>]*>)',
        re.IGNORECASE,
    )

    def typed_video(match):
        media_type = MEDIA_TYPES[Path(urlsplit(match[3]).path).suffix.lower()]
        return f'{match[1]}{match[4]}<source src={match[2]}{match[3]}{match[2]} type="{media_type}">'

    for path in tracked:
        if path.suffix not in TEXT_TYPES or path.parts[0] not in {"content", "layouts", "data", "assets"}:
            continue
        original = path.read_text()
        updated = replace_links(original, absolute)
        updated = replace_links(updated, relative.get(Path(*path.parts[:3]), {}))
        updated = video_tag.sub(typed_video, updated)
        if updated != original:
            path.write_text(updated)


def verify_build(before, after, urls):
    before_files, before_pages = before
    after_files, after_pages = after
    removed = {Path(path.lstrip("/")) for path in urls}
    if before_files - removed != after_files:
        raise RuntimeError("Migration changed other Hugo output files. Review these media references manually.")
    for page, links in before_pages.items():
        expected = collections.Counter()
        for url, count in links.items():
            parsed = urlsplit(url)
            target = urls.get(unquote(parsed.path)) if parsed.hostname == "haoailab.com" else None
            if target:
                url = urlsplit(target)._replace(query=parsed.query, fragment=parsed.fragment).geturl()
            expected[url] += count
        if expected != after_pages[page]:
            raise RuntimeError(f"Migration changed unexpected media references in {page}")


def release_assets(release_id):
    pages = api(f"releases/{release_id}/assets?per_page=100", "--paginate", "--slurp")
    return {asset["name"]: asset for page in pages for asset in page}


def verify_asset(remote, local):
    if (
        remote["state"] != "uploaded"
        or remote["size"] != local.size
        or remote["digest"] != f"sha256:{local.sha256}"
        or remote["content_type"] != MEDIA_TYPES[local.path.suffix.lower()]
    ):
        raise RuntimeError(f"Release verification failed for {local.name}")


def publish_release(tag, base, staged, assets):
    pages = api("releases?per_page=100", "--paginate", "--slurp")
    release = next((r for page in pages for r in page if r["tag_name"] == tag), None)
    if release is None:
        release = api(
            "releases", "--method", "POST", "-f", f"tag_name={tag}",
            "-f", f"target_commitish={base}", "-f", f"name=Website blog media {tag.removeprefix('blog-media-')}",
            "-f", "body=Original files embedded in posts on haoailab.com.",
            "-F", "draft=true", "-F", "prerelease=true", "-f", "make_latest=false",
        )
    by_name = {asset.name: asset for asset in assets}
    remote_assets = release_assets(release["id"])
    if remote_assets.keys() - by_name.keys():
        raise RuntimeError(f"Unexpected files already exist in {tag}")
    for name, remote in remote_assets.items():
        if remote["state"] == "starter" and release["draft"]:
            api(f"releases/assets/{remote['id']}", "--method", "DELETE")
        else:
            verify_asset(remote, by_name[name])
    missing = [staged / name for name in by_name if remote_assets.get(name, {}).get("state") != "uploaded"]
    if missing:
        if not release["draft"]:
            raise RuntimeError(f"Published release {tag} is missing files")
        command("gh", "release", "upload", tag, *map(str, missing), "--repo", REPO)
    remote_assets = release_assets(release["id"])
    for name, asset in by_name.items():
        verify_asset(remote_assets[name], asset)
    if release["draft"]:
        command("gh", "release", "edit", tag, "--draft=false", "--prerelease", "--latest=false", "--repo", REPO)
    for asset in by_name.values():
        url = f"https://github.com/{REPO}/releases/download/{tag}/{asset.name}"
        verify_public_file(url, staged / asset.name)


def verify_public_file(url, original):
    response = subprocess.check_output([
        "curl", "--fail", "--silent", "--show-error", "--location", "--retry", "2",
        "--max-time", "30", "--max-filesize", "1048576", "--range", "0-31",
        "--write-out", "\n%{http_code}", url,
    ])
    with original.open("rb") as file:
        if response != file.read(32) + b"\n206":
            raise RuntimeError(f"Public range verification failed for {url}")


def open_pr(tag, base, assets):
    branch = BRANCH_PREFIX + tag.removeprefix("blog-media-")
    existing = json.loads(command("gh", "pr", "list", "--repo", REPO, "--head", branch,
                                  "--state", "all", "--json", "url"))
    if existing:
        print(f"Migration PR already exists: {existing[0]['url']}")
        return
    if api(f"branches/{DEFAULT_BRANCH}")["commit"]["sha"] != base:
        raise RuntimeError(f"{DEFAULT_BRANCH} changed during verification. The next run will retry.")
    command("git", "add", "--update", "--", "content", "layouts", "data", "assets")
    remote_branch = command("git", "ls-remote", "--heads", "origin", branch)
    if remote_branch:
        command("git", "fetch", "--depth=2", "origin", branch)
        changed = command("git", "diff", "--cached", "--name-only").splitlines()
        remote_changed = command("git", "diff", "--name-only", "FETCH_HEAD^", "FETCH_HEAD").splitlines()
        if changed != remote_changed or command("git", "diff", "--cached", "--name-only", "FETCH_HEAD", "--", *changed):
            raise RuntimeError(f"Existing branch {branch} contains different changes")
    else:
        command("git", "-c", "user.name=github-actions[bot]",
                "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
                "commit", "-m", "Archive settled blog media")
        command("git", "push", "origin", f"HEAD:refs/heads/{branch}")
    body = (
        f"Moves {len(assets)} original media files ({sum(a.size for a in assets):,} bytes) to "
        f"the [verified release](https://github.com/{REPO}/releases/tag/{tag}). "
        "Hugo output links, SHA-256 digests, and anonymous byte-range responses passed verification."
    )
    env = dict(os.environ, GH_TOKEN=os.environ["MEDIA_ARCHIVE_PR_TOKEN"])
    print(command("gh", "pr", "create", "--repo", REPO, "--base", DEFAULT_BRANCH, "--head", branch,
                  "--title", "Archive settled blog media", "--body", body, env=env))


def main():
    dry_run = os.environ.get("DRY_RUN") == "true"
    if not dry_run and not os.environ.get("MEDIA_ARCHIVE_PR_TOKEN"):
        print("::notice::Set the MEDIA_ARCHIVE_PR_TOKEN repository secret to enable migration PRs.")
        return
    if command("git", "status", "--porcelain"):
        raise RuntimeError("Run from a clean checkout")
    prs = json.loads(command("gh", "pr", "list", "--repo", REPO, "--limit", "100", "--json", "headRefName,url"))
    if pending := next((pr for pr in prs if pr["headRefName"].startswith(BRANCH_PREFIX)), None):
        print(f"Media PR awaiting review: {pending['url']}")
        return
    base = command("git", "rev-parse", "HEAD")
    tracked = {}
    for record in command("git", "ls-files", "--stage", "-z").split("\0"):
        if record:
            metadata, path = record.split("\t", 1)
            tracked[Path(path)] = metadata.split()[1]
    settled = list(candidates(tracked, dt.datetime.now(dt.timezone.utc)))
    if not settled:
        print("No blog media has reached the seven-day cutoff.")
        return
    with tempfile.TemporaryDirectory(prefix="blog-media-") as temp:
        scratch = Path(temp)
        before = build(scratch / "before", scratch / "resources")
        references = {
            unquote(urlsplit(url).path)
            for page in before[1].values() for url in page
            if urlsplit(url).hostname == "haoailab.com"
        }
        assets = [asset for asset in settled if asset.public_path in references]
        if not assets:
            print("No settled files have a direct media reference in the rendered site.")
            return
        names = {a.name for a in assets}
        if len(names) > 1000:
            raise RuntimeError("One GitHub release supports up to 1,000 distinct assets")
        if len({a.public_path for a in assets}) != len(assets):
            raise RuntimeError("Multiple media files map to the same published URL")
        signature = hashlib.sha256("\n".join(f"{a.public_path}:{a.sha256}" for a in assets).encode())
        tag = f"blog-media-{signature.hexdigest()[:16]}"
        prefix = f"https://github.com/{REPO}/releases/download/{tag}/"
        urls = {asset.public_path: prefix + asset.name for asset in assets}
        staged = scratch / "assets"
        staged.mkdir()
        for asset in assets:
            target = staged / asset.name
            if not target.exists():
                shutil.copyfile(asset.path, target)
        edit_sources(tracked, assets, urls)
        for asset in assets:
            asset.path.unlink()
        verify_build(before, build(scratch / "after", scratch / "resources"), urls)
        print(f"Verified the Hugo migration for {len(assets)} media references.")
        if dry_run:
            return
        publish_release(tag, base, staged, assets)
        open_pr(tag, base, assets)


if __name__ == "__main__":
    main()
