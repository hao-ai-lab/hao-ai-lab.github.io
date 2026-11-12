# [hao-ai-lab.github.io](https://hao-ai-lab.github.io/)

This git repo hosts all content displayed on the website of Hao AI Lab @ UCSD.


## How to build

1. Install Hugo > 0.12
2. Run the following commands:
```bash
git clone https://github.com/hao-ai-lab/hao-ai-lab.github.io.git
cd hao-ai-lab.github.io
hugo server
```


## Blog demo media

Large blog GIFs and MP4s are hosted in
[demo media v1](https://github.com/hao-ai-lab/hao-ai-lab.github.io/releases/tag/website-demo-media-v1)
and [demo media v2](https://github.com/hao-ai-lab/hao-ai-lab.github.io/releases/tag/website-demo-media-v2).
Keep these releases available while the posts link to their files.

New posts can keep GIF, MP4, and WebM files in their page bundle during launch
week. The daily **Archive settled blog media** workflow checks whether the same
post and files appeared in a successful Pages deploy at least seven days ago.
It opens a migration PR after validating the rendered links, release checksums,
and public byte-range responses. A maintainer merges the PR to publish the new
links. Media with custom references that fail Hugo validation stays in the
site until it can be migrated separately.

Run the workflow manually with **dry run** selected to check a migration without
uploading files or opening a PR. The media workflow is separate from the normal
Pages deploy. GitHub's organization policy requires a separate credential to
open the PR. Set the repository Actions secret `MEDIA_ARCHIVE_PR_TOKEN` to a
fine-grained bot token scoped to this repository with **Pull requests: Read and
write**. The built-in Actions token stages releases and pushes the migration
branch. With the PR token absent, scheduled runs skip uploading and dry runs
remain available.

## How to generate publication list

1. Copy the latest publication json to the root folder.
2. Run the following
```bash
python gen_publications.py
```
3. Copy the output of this script to `content/publications.md`