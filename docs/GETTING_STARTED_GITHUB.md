# First push to GitHub (personal reminder)

1. Create an empty repository at https://github.com/new — name: `airwaylab`,
   visibility: **Private**, no README (this repo already has one).
2. On your computer, inside the extracted `airwaylab` folder:

```bash
cd airwaylab
git init
git add .
git commit -m "AirwayLab v1.0.0 — clean baseline"
git branch -M main
git remote add origin https://github.com/seccopower/airwaylab.git
git push -u origin main
```

3. The CI (synthetic-phantom tests) runs automatically on the first push.
4. If you later publish a release and want a citable DOI, connect the
   repository on https://zenodo.org and cut a GitHub release.

Privacy note: the repository does not and must never contain patient data —
only code, documentation, and synthetic tests. Anonymized NIfTI files stay out
of version control (`.gitignore`).
