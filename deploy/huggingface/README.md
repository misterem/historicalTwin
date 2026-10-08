---
title: Historical Twin API
emoji: 🖼️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 8080
pinned: false
short_description: Matches a selfie to the most similar portrait painting
---

# Historical Twin API

The matching API behind [twin.maxaltman.com](https://twin.maxaltman.com). Send a photo to
`POST /match` and get back the portrait paintings whose faces look most like it.

Photos are processed in memory and never stored.

Source code: https://github.com/misterem/historicalTwin. This Space is published from that
repo by `deploy/publish.py`, so change the code there rather than here.

Face models: InsightFace `buffalo_l` (non-commercial research use only). Paintings from WikiArt.
