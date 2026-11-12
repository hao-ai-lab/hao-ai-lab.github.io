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
Keep both releases available while the posts link to their files. For new large
demos, upload the original files to a versioned release and use the asset
download URLs in the post. Check playback and downloads in a browser before
deploying. The Pages workflow checks the generated site against the 1 GB limit.

## How to generate publication list

1. Copy the latest publication json to the root folder.
2. Run the following
```bash
python gen_publications.py
```
3. Copy the output of this script to `content/publications.md`