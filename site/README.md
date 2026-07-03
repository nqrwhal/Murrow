# Murrow Site

Astro frontend for Murrow. The site is intended to be fully static and consume
JSON produced by the pipeline publish stage.

## Commands

```bash
npm install
npm run dev
npm run build
npm run preview
```

## Data Seam

The pipeline will eventually publish small build-time imports into
`src/data/` and heavier event payloads into `public/data/events/`. Until those
artifacts exist, the site renders the project landing page and methodology
preview.
