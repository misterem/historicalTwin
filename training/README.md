# training (not started)

Only worth doing if the pretrained ArcFace baseline gives poor matches. The plan is to freeze the
embedder and train a small projection head with a contrastive loss on photo↔"painted" pairs:
photo faces (e.g. FFHQ) paired with stylized versions of themselves. Run it on a cloud GPU.
The output is just a new `embeddings.npy`, so the backend doesn't change.
