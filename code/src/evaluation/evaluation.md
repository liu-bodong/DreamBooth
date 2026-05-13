# Evaluation metrics

The paper uses the following metrics:

- DINO: the average pairwise cosine similarity between the ViT-S/16 DINO embeddings of generated and real images (page 6)

- CLIP-I:

- CLIP-T: average cosine similarity between prompt and image CLIP embeddings.

---

- Subject Fidelity: CLIP-I / DINO

- Prompt Fidelity: CLIP-T

---

- PRES (Prior Preservation): computed by the average pairwise DINO embeddings between generated images of random subjects of the prior class and real images of our specific subject (page 7). Lower is better.

- DIV (Diversity): computed by using the average LIPIS cosine similarity between generated images of same subject with the same prompt (page 7).

---

